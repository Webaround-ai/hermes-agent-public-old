"""Durable HTTP withdrawal refuses the next native action without killing an in-flight call."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from agent.cooperative_stop import (install_cooperative_stop, stop_at_iteration_boundary,
                                    note_action_started, stopped_before_action)
from agent.tool_executor import _pre_tool_block
from cron.jobs import create_job, get_job, use_cron_store
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


def test_http_stop_persists_and_native_admission_waits_for_inflight_action(tmp_path):
    async def go():
        with use_cron_store(tmp_path):
            job = create_job(prompt='Fake authorized work.', schedule='0 9 * * *', name='stop-test')
            adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'fake-api-key'}))
            app = web.Application()
            app.router.add_post('/api/jobs/{job_id}/stop', adapter._handle_stop_job)
            agent = SimpleNamespace(_interrupt_requested=False)
            install_cooperative_stop(agent, lambda: bool((get_job(job['id']) or {}).get('stop_requested')))
            admitted = threading.Event()
            finish_action = threading.Event()
            effects = []
            def already_started():
                admitted.set()
                assert finish_action.wait(5)
                effects.append('confirmed fake payment')
            note_action_started(agent)
            with ThreadPoolExecutor(1) as pool:
                running = pool.submit(already_started)
                assert admitted.wait(2)
                async with TestClient(TestServer(app)) as client:
                    denied = await client.post('/api/jobs/' + job['id'] + '/stop')
                    assert denied.status == 401
                    response = await client.post('/api/jobs/' + job['id'] + '/stop',
                        headers={'Authorization': 'Bearer fake-api-key'})
                    assert response.status == 200, await response.text()
                    stopped = (await response.json())['job']
                    assert stopped['enabled'] is False and stopped['stop_requested'] is True
                    # Admission refuses every new action but does not signal the running tool waiter.
                    ref = SimpleNamespace(name='fake_send', args={}, task_id='fake', call_id='fake', trace=[])
                    reason, args = _pre_tool_block(agent, ref)
                    assert 'not started' in reason and args == {}
                    assert agent._interrupt_requested is False and not running.done()
                    finish_action.set()
                    running.result(timeout=2)
                    assert effects == ['confirmed fake payment']
                    assert stopped_before_action(agent) is False
                    assert stop_at_iteration_boundary(agent) is True and agent._interrupt_requested is True
                # A new agent after restart reads the same durable withdrawal.
                fresh = SimpleNamespace(_interrupt_requested=False)
                install_cooperative_stop(fresh, lambda: bool(get_job(job['id']).get('stop_requested')))
                assert stop_at_iteration_boundary(fresh) is True
    asyncio.run(go())


def test_cancelled_report_retains_late_outcome_idempotently(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('origin', source='api_server', session_key='owner:fake')
        kwargs = dict(task_id='task-fake', generation='1', title='Booking', state='cancelled')
        first = db.append_iollo_task_report('origin', 'Execution ended. Booking confirmed: fake-42.', **kwargs)
        duplicate = db.append_iollo_task_report('origin', 'Execution ended. Booking confirmed: fake-42.', **kwargs)
        assert first[:2] == duplicate[:2] and first[2] is True and duplicate[2] is False
        assert db.get_messages('origin')[-1]['display_metadata']['state'] == 'cancelled'
    finally:
        db.close()



def test_only_native_stop_before_any_action_proves_safe_continuation():
    cold = SimpleNamespace(_interrupt_requested=False)
    install_cooperative_stop(cold, lambda: True)
    assert stopped_before_action(cold) is False
    assert stop_at_iteration_boundary(cold) is True
    assert stopped_before_action(cold) is True
    note_action_started(cold)
    assert stopped_before_action(cold) is False


def test_unavailable_withdrawal_check_blocks_work_without_claiming_owner_withdrawal():
    agent = SimpleNamespace(_interrupt_requested=False)
    def unavailable():
        raise OSError('Synthetic authority store unavailable')
    install_cooperative_stop(agent, unavailable)
    blocked, _ = _pre_tool_block(agent, SimpleNamespace(args={}))
    assert blocked and 'not started' in blocked
    assert stop_at_iteration_boundary(agent) is True
    assert 'Owner requested' not in agent._interrupt_message
    assert stopped_before_action(agent) is False  # failure is not authoritative proof for automatic continuation



def test_native_safe_stop_proof_is_durable_and_distinct_from_model_error(tmp_path, monkeypatch):
    from cron import executions
    monkeypatch.setattr(executions, 'EXECUTIONS_FILE', tmp_path / 'executions.db')
    first = executions.create_execution('fake-job', source='direct')
    forged = executions.finish_execution(first['id'], success=False,
        error='Owner withdrawal confirmed before any action started.')
    assert forged['stopped_before_action'] == 0
    second = executions.create_execution('fake-job', source='direct')
    assert executions.confirm_stop_before_action(second['id']) is True
    confirmed = executions.finish_execution(second['id'], success=False, error='Owner stopped.')
    assert confirmed['stopped_before_action'] == 1
    assert executions.confirm_stop_before_action(first['id']) is False



def test_delegated_child_inherits_withdrawal_without_killing_its_inflight_action():
    from tools.delegate_tool_child_run import _attach_child
    withdrawn = threading.Event()
    parent = SimpleNamespace(_active_children=[], _interrupt_requested=False)
    install_cooperative_stop(parent, withdrawn.is_set)
    child = SimpleNamespace(_interrupt_requested=False)
    _attach_child(parent, child)
    grandchild = SimpleNamespace(_interrupt_requested=False)
    _attach_child(child, grandchild)
    started, finish = threading.Event(), threading.Event()
    effects = []
    def active():
        note_action_started(child)
        started.set()
        assert finish.wait(3)
        effects.append('confirmed fake submission')
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(active)
        assert started.wait(2)
        withdrawn.set()
        ref = SimpleNamespace(name='fake_send', args={}, task_id='fake', call_id='fake', trace=[])
        assert 'not started' in _pre_tool_block(child, ref)[0]
        assert 'not started' in _pre_tool_block(grandchild, ref)[0]
        assert not future.done() and child._interrupt_requested is False
        finish.set()
        future.result(timeout=2)
        assert effects == ['confirmed fake submission']
        assert stop_at_iteration_boundary(child) is True
        assert stopped_before_action(child) is False


def test_explicit_native_resume_clears_withdrawal_for_future_runs(tmp_path):
    from cron.jobs import resume_job, update_job
    with use_cron_store(tmp_path):
        job = create_job(prompt='Fake work.', schedule='every 2h', name='resume-test')
        update_job(job['id'], {'state': 'paused', 'enabled': False, 'stop_requested': True})
        resumed = resume_job(job['id'])
        assert resumed['enabled'] is True and resumed['stop_requested'] is False


@pytest.mark.parametrize('text,silent,cleaned', [
    ('[SILENT]', True, ''),
    ('NO_REPLY', True, ''),
    ('[SILENT] No changes detected.', True, 'No changes detected.'),
    ('[SILENT]\nNo changes detected.', True, 'No changes detected.'),
    ('No changes detected.\nNO_REPLY', True, 'No changes detected.'),
    ('The guide mentions [SILENT] as a token.', False, 'The guide mentions [SILENT] as a token.'),
    ('[SILENT] Provider refused the action.', True, 'Provider refused the action.'),
])
def test_task_output_http_uses_native_canonical_silence_without_modifying_history(tmp_path, text, silent, cleaned):
    async def go():
        db = SessionDB(tmp_path / 'state.db')
        db.create_session('cron_fake_output', source='cron')
        db.append_message('cron_fake_output', 'user', 'Check for changes.')
        db.append_message('cron_fake_output', 'assistant', text)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'fake-api-key'}))
        adapter._session_db = db
        app = web.Application()
        app.router.add_get('/api/sessions/{session_id}/task-output', adapter._handle_iollo_task_output)
        try:
            async with TestClient(TestServer(app)) as client:
                assert (await client.get('/api/sessions/cron_fake_output/task-output')).status == 401
                response = await client.get('/api/sessions/cron_fake_output/task-output?order=latest&limit=40',
                    headers={'Authorization': 'Bearer fake-api-key'})
                assert response.status == 200, await response.text()
                assert await response.json() == {'text': cleaned, 'silent': silent}
                missing = await client.get('/api/sessions/missing/task-output',
                    headers={'Authorization': 'Bearer fake-api-key'})
                assert missing.status == 404
            assert db.get_messages('cron_fake_output')[-1]['content'] == text
        finally:
            db.close()
    asyncio.run(go())



@pytest.mark.parametrize('wait_at', ['terminal_approval', 'dispatch_order'])
def test_withdrawal_while_queued_is_rechecked_at_actual_native_dispatch(monkeypatch, wait_at):
    from agent import tool_executor
    from agent import terminal_approval_batch
    withdrawn, waiting, release = threading.Event(), threading.Event(), threading.Event()
    effects, posts = [], []
    agent = SimpleNamespace(_interrupt_requested=False,
        _tool_guardrails=SimpleNamespace(before_call=lambda *args: SimpleNamespace(allows_execution=True)))
    install_cooperative_stop(agent, withdrawn.is_set)
    ref = SimpleNamespace(name='fake_payment', args={}, emit_post=lambda *args, **kw: posts.append(kw))
    state = SimpleNamespace(blocked=False, args={})
    monkeypatch.setattr(tool_executor, '_pre_tool_block', lambda agent, ref: (None, ref.args))
    monkeypatch.setattr(tool_executor, '_begin_tool_execution', lambda *args: None)
    def wait():
        waiting.set()
        assert release.wait(3)
    monkeypatch.setattr(terminal_approval_batch, 'prepare_current_terminal',
        lambda ref: wait() if wait_at == 'terminal_approval' else None)
    def admission(callback=None):
        if wait_at == 'dispatch_order':
            wait()
        if callback:
            callback()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(tool_executor._dispatch_authorized_once, agent, state, ref,
            execute=lambda args: effects.append('payment'), scope_block=None,
            display_index=None, begin_execution=admission, authorization_gate=None)
        assert waiting.wait(2)
        withdrawn.set()
        release.set()
        assert 'not started' in future.result(timeout=2)
    assert effects == [] and state.blocked is True
    assert posts[-1]['error_type'] == 'owner_stop'
    assert agent._interrupt_requested is False and stopped_before_action(agent) is True
