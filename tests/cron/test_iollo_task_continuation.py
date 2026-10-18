"""Native task continuation preserves tool evidence and admits one scheduler occurrence."""
import pytest

from cron import executions, jobs, scheduler
from hermes_state import SessionDB


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(jobs, 'CRON_DIR', tmp_path / 'cron')
    monkeypatch.setattr(jobs, 'JOBS_FILE', tmp_path / 'cron/jobs.json')
    monkeypatch.setattr(jobs, 'OUTPUT_DIR', tmp_path / 'cron/output')
    monkeypatch.setattr(executions, 'EXECUTIONS_FILE', tmp_path / 'cron/executions.db')
    return tmp_path


def terminal_job():
    job = jobs.create_job('Complete authorized booking.', 'in 30m', deliver='local')
    attempt = executions.create_execution(job['id'], source='builtin')
    executions.mark_execution_running(attempt['id'])
    executions.bind_execution_session(attempt['id'], 'original-task-context')
    executions.confirm_continuation_ready(attempt['id'])
    executions.finish_execution(attempt['id'], success=False, error='Owner revised instruction after tool.')
    jobs.mark_job_run(job['id'], False, 'Owner revised instruction after tool.')
    return job, attempt


def test_continuation_admission_survives_lost_reply_without_duplicate_occurrence(native):
    job, attempt = terminal_job()
    first = jobs.continue_job(job['id'], attempt['id'], 'handoff-1', 'Only mornings. Inspect prior result.')
    assert first['manual_resume']['session_id'] == 'original-task-context'
    assert first['stop_requested'] is False
    next_run = first['next_run_at']
    duplicate = jobs.continue_job(job['id'], attempt['id'], 'handoff-1', 'Only mornings. Inspect prior result.')
    assert duplicate['next_run_at'] == next_run
    jobs.mark_job_run(job['id'], True)
    again = jobs.continue_job(job['id'], attempt['id'], 'handoff-1', 'Only mornings. Inspect prior result.')
    assert again['enabled'] is False and again['state'] == 'completed'
    assert not again.get('manual_resume')
    with pytest.raises(ValueError, match='conflicts'):
        jobs.continue_job(job['id'], attempt['id'], 'handoff-1', 'Changed content')


def test_live_or_unidentified_execution_cannot_be_continued(native):
    job = jobs.create_job('Work.', 'in 30m')
    attempt = executions.create_execution(job['id'], source='builtin')
    with pytest.raises(ValueError, match='terminal native context'):
        jobs.continue_job(job['id'], attempt['id'], 'handoff-1', 'Continue')
    executions.finish_execution(attempt['id'], success=False)
    with pytest.raises(ValueError, match='terminal native context'):
        jobs.continue_job(job['id'], attempt['id'], 'handoff-1', 'Continue')


def test_native_resume_restores_tool_outcome_at_compression_tip_and_same_run_path(native, monkeypatch):
    db = SessionDB(native / 'state.db')
    db.create_session('original-task-context', source='cron')
    db.create_session('compressed-tip', source='cron', parent_session_id='original-task-context')
    # Exercise the native lineage resolver while leaving this fixture independent of compression machinery.
    monkeypatch.setattr(db, 'get_compression_tip', lambda session_id: 'compressed-tip')
    db.append_message('compressed-tip', 'user', 'Book the authorized flight.')
    db.append_message('compressed-tip', 'assistant', tool_calls=[{
        'id': 'booking-1', 'type': 'function', 'function': {'name': 'book', 'arguments': '{}'}}])
    db.append_message('compressed-tip', 'tool', 'Confirmed booking fake-42.', tool_call_id='booking-1', tool_name='book')
    db.end_session('compressed-tip', 'interrupted')
    target, history = scheduler._resume_cron_context(db, {'session_id': 'original-task-context'})
    assert target == 'compressed-tip'
    assert any(m.get('role') == 'tool' and 'fake-42' in m.get('content', '') for m in history)
    assert db.get_session(target)['ended_at'] is None
    actions = []
    class Agent:
        def run_conversation(self, prompt, *, task_id, conversation_history):
            receipt = next(m['content'] for m in conversation_history if m.get('tool_call_id') == 'booking-1')
            assert 'fake-42' in receipt and prompt == 'Only mornings. Inspect prior booking before retry.'
            actions.append('inspect existing fake-42')
            return {'final_response': 'Existing booking inspected; no duplicate booking.'}
    monkeypatch.setattr(scheduler, '_cron_inactivity_seconds', lambda: 0)
    scheduler._run_agent_with_watchdog(Agent(), 'Only mornings. Inspect prior booking before retry.',
                                     {}, 'job-1', 'Task', '', None, conversation_history=history)
    assert actions == ['inspect existing fake-42']
    monkeypatch.setattr(db, 'get_compression_tip', lambda session_id: session_id)
    with pytest.raises(RuntimeError, match='missing|not found'):
        scheduler._resume_cron_context(db, {'session_id': 'missing'})
    db.close()


def test_unknown_live_owner_cannot_be_resumed_even_with_context(native, monkeypatch):
    job, attempt = terminal_job()
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET status='unknown',continuation_ready=0 WHERE id=?", (attempt['id'],))
    monkeypatch.setattr(executions, '_owner_is_live', lambda *_: True)
    with pytest.raises(ValueError, match='terminal native context'):
        jobs.continue_job(job['id'], attempt['id'], 'resume-unknown', 'Inspect before retry.')
    monkeypatch.setattr(executions, '_owner_is_live', lambda *_: False)
    accepted = jobs.continue_job(job['id'], attempt['id'], 'resume-unknown', 'Inspect before retry.')
    assert accepted['manual_resume']['execution_id'] == attempt['id']


def test_authenticated_existing_run_endpoint_admits_exact_continuation_once(native):
    import asyncio
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    job, attempt = terminal_job()
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'fake-owner-box-key'}))
    async def run():
        app = web.Application()
        app.router.add_post('/api/jobs/{job_id}/run', adapter._handle_run_job)
        async with TestClient(TestServer(app)) as client:
            payload = {'prompt': 'Inspect prior receipt; use revised direction.',
                       'resume_execution_id': attempt['id'], 'request_id': 'owner-handoff'}
            route = f'/api/jobs/{job["id"]}/run'
            assert (await client.post(route, json=payload)).status == 401
            headers = {'Authorization': 'Bearer fake-owner-box-key'}
            first = await client.post(route, json=payload, headers=headers)
            assert first.status == 200
            first_job = (await first.json())['job']
            second = await client.post(route, json=payload, headers=headers)
            assert second.status == 200
            second_job = (await second.json())['job']
            assert second_job['next_run_at'] == first_job['next_run_at']
            assert second_job['manual_resume']['session_id'] == 'original-task-context'
    asyncio.run(run())


def test_failed_attempt_cannot_continue_while_native_worker_is_unfinished(native, monkeypatch):
    job, attempt = terminal_job()
    with executions._transaction() as conn:
        conn.execute('UPDATE executions SET continuation_ready=0 WHERE id=?', (attempt['id'],))
    monkeypatch.setattr(executions, '_owner_is_live', lambda *_: True)
    with pytest.raises(ValueError, match='terminal native context'):
        jobs.continue_job(job['id'], attempt['id'], 'waiting-on-worker', 'Inspect prior outcome.')
    executions.confirm_continuation_ready(attempt['id'])
    assert jobs.continue_job(job['id'], attempt['id'], 'waiting-on-worker', 'Inspect prior outcome.')['enabled']


def test_late_replay_of_old_direction_never_replaces_new_direction(native):
    job, attempt = terminal_job()
    jobs.continue_job(job['id'], attempt['id'], 'direction-a', 'Direction A.')
    second = jobs.continue_job(job['id'], attempt['id'], 'direction-b', 'Direction B.')
    replay = jobs.continue_job(job['id'], attempt['id'], 'direction-a', 'Direction A.')
    assert replay['manual_resume']['request_id'] == 'direction-b'
    assert replay['next_run_at'] == second['next_run_at']


def test_execution_output_never_replays_an_earlier_attempt_assistant(native):
    import asyncio
    import json
    from datetime import datetime
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    job, attempt = terminal_job()
    record = executions.get_execution(attempt['id'])
    boundary = datetime.fromisoformat(record['started_at']).timestamp()
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'fake-owner-box-key'}))
    messages = [{'role': 'assistant', 'content': 'Prior attempt succeeded.', 'timestamp': boundary - 10}]
    async def session_messages(_request):
        return web.json_response({'data': messages})
    adapter._handle_session_messages = session_messages
    async def run():
        app = web.Application()
        app.router.add_get('/api/sessions/{session_id}/task-output', adapter._handle_iollo_task_output)
        async with TestClient(TestServer(app)) as client:
            route = f'/api/sessions/original-task-context/task-output?execution_id={attempt["id"]}'
            headers = {'Authorization': 'Bearer fake-owner-box-key'}
            response = await client.get(route, headers=headers)
            assert response.status == 200
            assert (await response.json()) == {'text': '', 'silent': False}
            messages.append({'role': 'assistant', 'content': 'Current attempt failed.', 'timestamp': boundary + 1})
            response = await client.get(route, headers=headers)
            assert (await response.json())['text'] == 'Current attempt failed.'
    asyncio.run(run())
