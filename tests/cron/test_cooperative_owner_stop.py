"""Durable HTTP withdrawal refuses the next native action without killing an in-flight call."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from agent.cooperative_stop import install_cooperative_stop, stop_at_iteration_boundary
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
