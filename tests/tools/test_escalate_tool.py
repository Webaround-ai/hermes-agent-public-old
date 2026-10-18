"""escalate: advice-only child on the configured tier (tools/escalate_tool.py).

Contracts: not advertised unless delegation.escalate names a configured tier; caps per task and per day
are enforced in code before launching the advisor; the advisor child is read-only
(no terminal/file/delegation/sends) on the
tier's model and effort; children can never escalate.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

import tools.delegate_tool as dt
import tools.delegate_tool_config as dtc
import tools.escalate_tool as et
from hermes_cli import plugins

CFG = {
    "tiers": {"opus_high": {"model": "anthropic/claude-opus-5.5", "reasoning_effort": "high"},
              "haiku_low": {"model": "anthropic/claude-haiku-4.5", "reasoning_effort": "low"}},
    "escalate": {"tier": "opus_high", "max_per_task": 2, "max_per_day": 3},
}
ARGS = dict(question="Which migration order is safe?", context="Tried A then B: 'FK violation on users.id'",
            constraints="no downtime", wanted="the order to run the migrations in")


@pytest.fixture(autouse=True)
def _reset():
    et._reset_counters_for_tests()
    yield
    et._reset_counters_for_tests()


def _use_cfg(monkeypatch, cfg):
    monkeypatch.setattr(dtc, "_load_config", lambda: cfg)
    monkeypatch.setattr(dt, "_load_config", lambda: cfg)


def _parent(turn="turn-1", depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-5"
    parent.reasoning_config = {"enabled": True, "effort": "low"}
    parent.request_overrides = None
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent.session_id = "sess-1"
    parent._current_turn_id = turn
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = ["terminal", "file", "web", "vision", "delegation", "messaging", "session_search",
                               "memory", "escalate"]
    parent.disabled_toolsets = []
    return parent


def _fake_advisor(monkeypatch):
    calls = []

    def run(parent_agent, goal, brief, route, base_creds, esc):
        calls.append({"goal": goal, "brief": brief, "model": route["model"]})
        return {"results": [{"status": "completed", "summary": "Run B first, then A.", "cost_usd": 0.01}]}

    monkeypatch.setattr(et, "_run_advisor", run)
    return calls


def test_not_advertised_unless_configured_with_a_known_tier(monkeypatch):
    for cfg, expected in (({}, False), ({**CFG, "escalate": {"tier": "missing"}}, False),
                          ({**CFG, "escalate": "yes"}, False), (CFG, True)):
        _use_cfg(monkeypatch, cfg)
        assert et.check_escalate_requirements() is expected


def test_caps_per_task_and_per_day_are_enforced_in_code(monkeypatch):
    _use_cfg(monkeypatch, CFG)
    calls = _fake_advisor(monkeypatch)
    turn1 = _parent("turn-1")
    first, second, third = (json.loads(et.escalate(**ARGS, parent_agent=turn1)) for _ in range(3))
    assert first["escalated"] and second["escalated"] and first["advice"] == "Run B first, then A."
    assert third["escalated"] is False and "this task" in third["reason"]
    turn2 = _parent("turn-2")
    assert json.loads(et.escalate(**ARGS, parent_agent=turn2))["escalated"] is True
    refused = json.loads(et.escalate(**ARGS, parent_agent=_parent("turn-3")))
    assert refused["escalated"] is False and "Daily" in refused["reason"]
    assert len(calls) == 3


def test_children_can_never_escalate(monkeypatch):
    from tools.delegate_tool_toolsets import _blocked_toolsets_for_role
    _use_cfg(monkeypatch, CFG)
    assert "escalate" in _blocked_toolsets_for_role("leaf")
    assert "escalate" in _blocked_toolsets_for_role("orchestrator")
    out = json.loads(et.escalate(**ARGS, parent_agent=_parent(depth=1)))
    assert "main agent" in out["error"]


def test_advisor_runs_read_only_on_the_tier_model_and_effort(monkeypatch):
    # Orchestration on: an ordinary child would get `delegation` back; the advisor must not.
    _use_cfg(monkeypatch, {**CFG, "orchestrator_enabled": True, "max_spawn_depth": 3,
                          "fallback_providers": [{"provider": "openrouter", "model": "cheap-model"}]})
    monkeypatch.setattr(dt, "_get_orchestrator_enabled", lambda: True)
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 3)
    parent = _parent()
    parent.enabled_toolsets.append("mcp-private")
    parent._fallback_chain = [{"provider": "openrouter", "model": "another-model"}]
    with patch("run_agent.AIAgent") as MockAgent:
        child = MagicMock()
        child.run_conversation.return_value = {"final_response": "Run B first, then A.", "completed": True,
                                               "api_calls": 2}
        MockAgent.return_value = child
        out = json.loads(et.escalate(**ARGS, parent_agent=parent))
    kwargs = MockAgent.call_args.kwargs
    assert out["escalated"] is True and out["advice"] == "Run B first, then A."
    assert (out["tier"], out["model"], out["reasoning_effort"]) == ("opus_high", "anthropic/claude-opus-5.5", "high")
    assert kwargs["model"] == "anthropic/claude-opus-5.5"
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert set(kwargs["enabled_toolsets"]) <= set(et.ESCALATE_READONLY_TOOLSETS)
    assert {"web", "vision", "session_search"} <= set(kwargs["enabled_toolsets"])
    assert "delegation" in kwargs["disabled_toolsets"]
    assert not kwargs["fallback_model"]
    assert parent.model == "anthropic/claude-sonnet-5"
    assert parent.reasoning_config == {"enabled": True, "effort": "low"}
    from model_tools import _select_tool_names
    advisor_tools = _select_tool_names(kwargs["enabled_toolsets"], kwargs["disabled_toolsets"], True)
    assert not ({"terminal", "execute_code", "read_file", "write_file", "send_message",
                 "delegate_task", "escalate", "cronjob_manage", "memory"} & advisor_tools)
    # The brief carries everything the cold advisor needs.
    prompt = kwargs["ephemeral_system_prompt"]
    for part in ARGS.values():
        assert part in prompt


# ── levels and optional caps (owner decisions, PR #11) ─────────────────────

LEVELS = {
    "tiers": {**CFG["tiers"], "fable": {"model": "anthropic/claude-fable-5.1", "reasoning_effort": "high"}},
    "escalate": {"tiers": ["opus_high", "fable"]},
}


def test_first_call_in_a_turn_uses_level_0_and_later_calls_level_1(monkeypatch):
    _use_cfg(monkeypatch, LEVELS)
    calls = _fake_advisor(monkeypatch)
    turn = _parent("turn-1")
    outs = [json.loads(et.escalate(**ARGS, parent_agent=turn)) for _ in range(3)]
    assert [o["advised_by"] for o in outs] == ["anthropic/claude-opus-5.5", "anthropic/claude-fable-5.1",
                                               "anthropic/claude-fable-5.1"]
    assert [o["level"] for o in outs] == [0, 1, 2]
    assert "anthropic/claude-opus-5.5" in outs[0]["note"]
    assert [c["model"] for c in calls] == [o["advised_by"] for o in outs]
    # A new turn starts again at level 0.
    assert json.loads(et.escalate(**ARGS, parent_agent=_parent("turn-2")))["tier"] == "opus_high"


def test_legacy_single_tier_key_is_a_one_level_list(monkeypatch):
    _use_cfg(monkeypatch, CFG)
    assert et.escalate_config()["tiers"] == ["opus_high"]
    _fake_advisor(monkeypatch)
    turn = _parent("turn-1")
    outs = [json.loads(et.escalate(**ARGS, parent_agent=turn)) for _ in range(2)]
    assert [o["tier"] for o in outs] == ["opus_high", "opus_high"]


def test_no_caps_by_default(monkeypatch):
    _use_cfg(monkeypatch, LEVELS)
    esc = et.escalate_config()
    assert (esc["max_per_task"], esc["max_per_day"]) == (None, None)
    calls = _fake_advisor(monkeypatch)
    turn = _parent("turn-1")
    outs = [json.loads(et.escalate(**ARGS, parent_agent=turn)) for _ in range(12)]
    assert all(o["escalated"] for o in outs) and len(calls) == 12


def test_caps_still_enforced_when_set_alongside_levels(monkeypatch):
    _use_cfg(monkeypatch, {**LEVELS, "escalate": {"tiers": ["opus_high", "fable"], "max_per_task": 1}})
    _fake_advisor(monkeypatch)
    turn = _parent("turn-1")
    assert json.loads(et.escalate(**ARGS, parent_agent=turn))["escalated"] is True
    refused = json.loads(et.escalate(**ARGS, parent_agent=turn))
    assert refused["escalated"] is False and "this task" in refused["reason"]


def test_direct_advice_ignores_obsolete_gate_and_never_replays_parent_upgrade(monkeypatch):
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    # Old persisted policy and installed callbacks cannot block or mutate direct advice.
    cfg = {**CFG, "escalate": {**CFG["escalate"], "reasoning_updates": True, "require_gate": True}}
    _use_cfg(monkeypatch, cfg)
    calls = _fake_advisor(monkeypatch)
    seen = []
    monkeypatch.setattr(plugins, "has_hook", lambda hook: seen.append(hook) or True)
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: ["think_harder"])
    parent = _parent()
    parent.model, parent.api_mode = "openai/gpt-6-luna", "codex_responses"
    result = json.loads(INLINE_TOOL_EXECUTORS["escalate"](parent, ARGS, InlineToolContext("task")))
    assert result["escalated"] is True and len(calls) == 1
    assert "escalate_gate" not in seen and "delegation_tier_chooser" not in seen
    assert parent.model == "openai/gpt-6-luna"
    assert parent.reasoning_config == {"enabled": True, "effort": "low"}
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "escalate", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"escalated": False,
         "reasoning_update": {"model": parent.model, "effort": "high"}})}]
    wire = _chat_messages_to_responses_input(messages, current_issuer_model="gpt-6-luna")
    assert not any(item.get("type") == "configuration_update" for item in wire)
