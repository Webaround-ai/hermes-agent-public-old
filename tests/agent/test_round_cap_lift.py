"""Iollo fork: a run's lowered round cap (POST /v1/runs max_iterations) lifts back once the run starts browser work."""
import json
from types import SimpleNamespace

from agent.iteration_budget import IterationBudget
from agent.turn_tool_round import _lift_round_cap_for_browser_work


def _agent(cap=8, lift_to=60):
    return SimpleNamespace(max_iterations=cap, iteration_budget=IterationBudget(cap), _round_cap_lift_to=lift_to)


def _call(name, arguments="{}"):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def test_lookups_keep_the_cap():
    agent = _agent()
    _lift_round_cap_for_browser_work(agent, [_call("search_answer"), _call("web_extract")])
    assert agent.max_iterations == 8 and agent.iteration_budget.max_total == 8 and agent._round_cap_lift_to == 60


def test_browser_work_lifts_the_cap_once():
    agent = _agent()
    agent.iteration_budget.consume()
    _lift_round_cap_for_browser_work(agent, [_call("search_answer"), _call("browser_exec")])
    assert agent.max_iterations == 60 and agent.iteration_budget.remaining == 59
    assert agent._round_cap_lift_to is None


def test_a_bridged_browser_tool_lifts_it_too(monkeypatch):
    import tools.tool_search as ts
    monkeypatch.setattr(ts, "resolve_underlying_call", lambda args: (args["calls"][0]["name"], {}, None))
    agent = _agent()
    _lift_round_cap_for_browser_work(agent, [_call("tool_call", json.dumps({"calls": [{"name": "browser_prepare"}]}))])
    assert agent.max_iterations == 60


def test_no_cap_means_nothing_to_lift():
    agent = _agent(cap=60, lift_to=None)
    _lift_round_cap_for_browser_work(agent, [_call("browser_exec")])
    assert agent.max_iterations == 60
