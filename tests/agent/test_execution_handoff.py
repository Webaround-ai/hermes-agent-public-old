"""Trusted immediate handoffs stop the native caller before another action is admitted."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_segmented
from agent.tool_dispatch_helpers import _plan_tool_batch_segments
from agent.tool_profile import set_ends_turn
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.registry import registry

HANDOFF = 'test_immediate_handoff'
WRITE = 'test_browser_write'
READ = 'test_safe_read'
ACK = 'The background task now owns this work.'
SUCCESS = json.dumps({'success': True, 'created': True, 'task': {'execution': {'schedule': None}}})


def transfer(args, result):
    data = json.loads(result)
    if (data.get('success') is True and not args.get('schedule')
            and data.get('task', {}).get('execution', {}).get('schedule') is None):
        return ACK
    return None


def call(name, cid, args=None):
    return SimpleNamespace(id=cid, type='function', function=SimpleNamespace(
        name=name, arguments=json.dumps(args or {})))


def response(calls=None, text=''):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=calls),
        finish_reason='tool_calls' if calls else 'stop')], model='fake/model', usage=None)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    schema = {'parameters': {'type': 'object', 'properties': {'schedule': {'type': 'object'}}}}
    for name in (HANDOFF, WRITE, READ):
        registry.register(name=name, toolset='handoff_test', schema={**schema, 'name': name},
            handler=lambda a, **k: '{}', execution_handoff=transfer if name == HANDOFF else None)
    defs = [{'type': 'function', 'function': {**schema, 'name': name, 'description': name}}
            for name in (HANDOFF, WRITE, READ)]
    from tools.tool_search import bridge_tool_schemas
    defs.extend(bridge_tool_schemas(3))
    monkeypatch.setattr('agent.title_generator.maybe_auto_title', lambda *a, **k: None)
    monkeypatch.setattr('agent.tool_executor._tool_search_scoped_names', lambda agent: {HANDOFF, WRITE, READ})
    with (patch('model_tools.get_tool_definitions', return_value=defs),
          patch('model_tools.check_toolset_requirements', return_value={}),
          patch('agent.process_bootstrap.OpenAI'), patch('run_agent._hermes_home', tmp_path),
          patch('agent.model_metadata.fetch_model_metadata', return_value={})):
        agent = AIAgent(api_key='fake-key', base_url='https://example.invalid', quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = 'A fake test assistant.'
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    set_ends_turn(agent, False)
    db = SessionDB(db_path=tmp_path / 'history.db')
    db.create_session(session_id='fake-handoff-session', source='api_server', model='fake/model')
    agent._session_db, agent._session_db_created, agent.session_id = db, True, 'fake-handoff-session'
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    yield agent, db
    db.close()
    for name in (HANDOFF, WRITE, READ):
        registry.deregister(name)


def run(harness, calls, *, result=SUCCESS, mode='normal', fail_persist=False, plain_result=None):
    agent, db = harness
    actions = []
    agent.client.chat.completions.create.side_effect = [response(calls), response(text='Normal continuation.')]
    def dispatch(name, args, task_id, **kwargs):
        actions.append(name)
        return result if name == HANDOFF else plain_result or json.dumps({'success': True})
    original = agent._execute_tool_calls
    def execute(message, messages, task_id, api_call_count=0):
        if mode == 'concurrent':
            return execute_tool_calls_concurrent(agent, message, messages, task_id, api_call_count)
        if mode == 'segmented':
            segments = [('sequential', message.tool_calls[:1]), ('parallel', message.tool_calls[1:])]
            return execute_tool_calls_segmented(agent, message, messages, task_id, api_call_count, segments=segments)
        return original(message, messages, task_id, api_call_count)
    flush = agent._flush_messages_to_session_db
    def durable(messages, *a, **k):
        if fail_persist and messages and messages[-1].get('role') == 'tool':
            return False
        return flush(messages, *a, **k)
    with (patch('model_tools.handle_function_call', side_effect=dispatch),
          patch.object(agent, '_execute_tool_calls', side_effect=execute),
          patch.object(agent, '_flush_messages_to_session_db', side_effect=durable),
          patch.object(agent, '_cleanup_task_resources'),
          patch.object(agent, '_save_trajectory'),
          patch('agent.conversation_loop._restore_or_build_system_prompt')):
        outcome = agent.run_conversation('Do the fake work.')
    return outcome, actions, db.get_messages_as_conversation(agent.session_id)


@pytest.mark.parametrize('broker', [False, True])
@pytest.mark.parametrize('mode', ['normal', 'concurrent', 'segmented'])
def test_successful_immediate_handoff_persists_and_skips_sibling_actions(harness, broker, mode):
    handoff = call('tool_call', 'handoff', {'calls': [{'name': HANDOFF, 'arguments': {}}]}) if broker else call(HANDOFF, 'handoff')
    outcome, actions, durable = run(harness, [handoff, call(WRITE, 'write'), call(READ, 'read')], mode=mode)
    assert actions == [HANDOFF], next((m for m in outcome['messages'] if m.get('role') == 'tool'), {})
    assert harness[0].client.chat.completions.create.call_count == 1
    assert outcome['completed'] is True and outcome['final_response'] == ACK
    assert outcome['turn_exit_reason'] == 'tool_ended_turn'
    paired = [row for row in durable if row.get('role') == 'tool']
    assert [row['tool_call_id'] for row in paired] == ['handoff', 'write', 'read']
    assert json.loads(paired[0]['content'])['created'] is True
    assert all('was not started' in row['content'] for row in paired[1:])
    assert durable[-1]['role'] == 'assistant' and durable[-1]['content'] == ACK


@pytest.mark.parametrize('args,result', [
    ({}, json.dumps({'success': False, 'error': 'Cannot start.'})),
    ({'schedule': {'kind': 'once', 'run_at': '2099-01-01T10:00:00Z'}}, SUCCESS),
    ({}, json.dumps({'success': True, 'created': True, 'task': {'execution': {'schedule': {'kind': 'once'}}}})),
])
def test_refused_or_scheduled_task_continues_normally(harness, args, result):
    outcome, actions, _ = run(harness, [call(HANDOFF, 'handoff', args), call(WRITE, 'write')], result=result)
    assert actions == [HANDOFF, WRITE]
    assert harness[0].client.chat.completions.create.call_count == 2
    assert outcome['final_response'] == 'Normal continuation.'


def test_failed_canonical_result_append_never_admits_a_sibling(harness):
    outcome, actions, _ = run(harness, [call(HANDOFF, 'handoff'), call(WRITE, 'write')], fail_persist=True)
    assert actions == [HANDOFF]
    assert outcome['turn_exit_reason'] == 'session_persistence_failed'
    assert harness[0]._tool_execution_handoff is None


def test_unregistered_result_marker_cannot_transfer_execution(harness):
    outcome, actions, _ = run(harness, [call(WRITE, 'write'), call(READ, 'read')],
        plain_result=json.dumps({'success': True, 'execution_handoff': 'Untrusted transfer marker'}))
    assert actions == [WRITE, READ]
    assert outcome['final_response'] == 'Normal continuation.'


def test_registered_handoff_is_a_barrier_even_for_otherwise_parallel_safe_tool(harness, monkeypatch):
    import agent.tool_dispatch_helpers as planner
    monkeypatch.setattr(planner, '_PARALLEL_SAFE_TOOLS', {HANDOFF, READ})
    planned = _plan_tool_batch_segments([call(HANDOFF, 'handoff'), call(READ, 'read')])
    assert planned[0][0] == 'sequential'
    assert planned[0][1][0].function.name == HANDOFF


def test_accepted_existing_immediate_task_also_transfers_execution(harness):
    accepted = json.dumps({'success': True, 'created': False, 'task': {'execution': {'schedule': None}}})
    outcome, actions, _ = run(harness, [call(HANDOFF, 'handoff'), call(WRITE, 'write')], result=accepted)
    assert actions == [HANDOFF]
    assert outcome['completed'] is True and outcome['final_response'] == ACK
    assert outcome['turn_exit_reason'] == 'tool_ended_turn'
    assert not harness[0]._interrupt_requested


def test_a_later_owner_turn_does_not_inherit_the_completed_handoff(harness):
    first, actions, _ = run(harness, [call(HANDOFF, 'first-handoff')])
    assert first['completed'] is True and actions == [HANDOFF]
    second, actions, _ = run(harness, [call(WRITE, 'next-turn-write')])
    assert actions == [WRITE] and second['final_response'] == 'Normal continuation.'
    assert harness[0]._tool_execution_handoff is None


def test_plugin_registration_forwards_the_trusted_handoff_callback():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    name = 'test_plugin_execution_handoff'
    manager = PluginManager(scope_key=registry.current_scope_key())
    context = PluginContext(PluginManifest(name='handoff_test_plugin', key='handoff_test_plugin'), manager)
    try:
        context.register_tool(name=name, toolset='handoff_test',
            schema={'name': name, 'parameters': {'type': 'object', 'properties': {}}},
            handler=lambda args, **kwargs: '{}', execution_handoff=transfer)
        assert registry.get_entry(name).execution_handoff is transfer
    finally:
        registry.deregister(name)
