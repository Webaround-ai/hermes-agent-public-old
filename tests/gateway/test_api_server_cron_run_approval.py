"""A cron run asks its owner through the live /v1/runs approval round trip (iollo task runs,
2026-10-14: a purchase run as a box task must reach the owner's card and resume on Approve).

The api_server adapter opens a bridge per cron run under run id = the cron session id: an
owner-required approval parks the run ``waiting_for_approval`` (the same ``_make_approval_notify``
a /v1/runs run uses) and ``POST /v1/runs/{run_id}/approval`` resumes it. Silence ends the wait at
the box's own approval timeout with a BLOCKED result the run can report. No bridge: still blocked.
"""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tools import approval as approval_mod
from tools import approval_context

RUN_ID = "cron_job1_20261014_192000"
AUTH = {"Authorization": "Bearer sk-secret"}


@pytest.fixture
def cron_env(monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    yield
    approval_mod.set_cron_run_bridge_provider(None)


def _app(adapter):
    app = web.Application()
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    return app


def _purchase_in_cron_run():
    """The cron run's thread: the bridge scope run_job opens, then the payment approval."""
    import cron.scheduler as sched

    with sched._cron_run_approval_bridge(RUN_ID):
        return approval_mod.request_tool_approval(
            "commit_purchase", "tier3:pay: commit:c-1234abcd",
            rule_key="iollo-tier3:pay:commit_purchase:c-1234abcd", require_human=True)


async def _status(cli):
    resp = await cli.get(f"/v1/runs/{RUN_ID}", headers=AUTH)
    return resp.status, (await resp.json() if resp.status == 200 else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, approved", [("approve", True), ("deny", False)])
async def test_cron_run_waits_for_the_owner_and_resumes_on_answer(cron_env, answer, approved):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    async with TestClient(TestServer(_app(adapter))) as cli:
        adapter._cron_bridge_loop = asyncio.get_running_loop()
        approval_mod.set_cron_run_bridge_provider(adapter._open_cron_run_bridge)
        run = asyncio.ensure_future(asyncio.to_thread(_purchase_in_cron_run))
        for _ in range(200):
            code, status = await _status(cli)
            if code == 200 and status["status"] == "waiting_for_approval":
                break
            await asyncio.sleep(0.02)
        assert status["status"] == "waiting_for_approval"
        assert status["approval"]["choices"] == ["once", "deny"]

        resp = await cli.post(f"/v1/runs/{RUN_ID}/approval", json={"choice": answer, "request_id": status["approval"]["request_id"]}, headers=AUTH)
        assert resp.status == 200
        result = await asyncio.wait_for(run, 10)
        assert result["approved"] is approved
        assert (await _status(cli))[0] == 404          # the run's bridge closed with the run


@pytest.mark.asyncio
async def test_owner_silence_ends_at_the_box_timeout(cron_env, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_timeout_for", lambda *_a: 1)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    async with TestClient(TestServer(_app(adapter))):
        adapter._cron_bridge_loop = asyncio.get_running_loop()
        approval_mod.set_cron_run_bridge_provider(adapter._open_cron_run_bridge)
        result = await asyncio.wait_for(asyncio.to_thread(_purchase_in_cron_run), 10)
    assert result["approved"] is False
    assert "timed out" in result["message"]


def test_cron_run_without_a_bridge_still_blocks(cron_env):
    approval_mod.set_cron_run_bridge_provider(None)
    result = _purchase_in_cron_run()
    assert result["approved"] is False
    assert "no interactive user or gateway" in result["message"]


@pytest.mark.asyncio
async def test_reused_cron_session_refuses_missing_and_previous_approval_identity(cron_env):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    async with TestClient(TestServer(_app(adapter))) as cli:
        adapter._cron_bridge_loop = asyncio.get_running_loop()
        approval_mod.set_cron_run_bridge_provider(adapter._open_cron_run_bridge)
        prior_request_id = None
        for attempt in range(2):
            run = asyncio.ensure_future(asyncio.to_thread(_purchase_in_cron_run))
            for _ in range(200):
                code, status = await _status(cli)
                if code == 200 and status["status"] == "waiting_for_approval":
                    break
                await asyncio.sleep(0.02)
            assert status["status"] == "waiting_for_approval"
            request_id = status["approval"]["request_id"]
            for body in ({"choice": "once"}, {"choice": "once", "resolve_all": True}):
                missing = await cli.post(f"/v1/runs/{RUN_ID}/approval", json=body, headers=AUTH)
                assert missing.status == 400
                assert (await missing.json())["error"]["code"] == "approval_request_required"
                assert not run.done()
            if attempt:
                assert request_id != prior_request_id
                stale = await cli.post(f"/v1/runs/{RUN_ID}/approval",
                                       json={"choice": "once", "request_id": prior_request_id}, headers=AUTH)
                assert stale.status == 409
                assert (await stale.json())["error"]["code"] == "approval_not_pending"
                assert not run.done()
            exact = await cli.post(f"/v1/runs/{RUN_ID}/approval",
                                   json={"choice": "deny", "request_id": request_id}, headers=AUTH)
            assert exact.status == 200 and (await exact.json())["request_id"] == request_id
            assert (await asyncio.wait_for(run, 10))["approved"] is False
            prior_request_id = request_id
