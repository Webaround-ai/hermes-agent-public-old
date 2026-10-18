"""Chooser hooks (hermes_cli/plugin_choices.py): the first accepted non-None answer wins, and anything that
goes wrong (no plugin, exception, timeout, off-list answer) is None, i.e. the configured default."""

import time

import pytest

from hermes_cli import plugins
from hermes_cli.plugin_choices import first_plugin_choice
from hermes_cli.plugins import VALID_HOOKS, PluginManager

_CHOOSERS = ("delegation_tier_chooser", "busy_input_chooser")


def _install(monkeypatch, hook, *callbacks):
    mgr = PluginManager()
    mgr._discovered = True
    mgr._hooks[hook] = list(callbacks)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    return mgr


def _upper(answer):
    return answer.upper() if isinstance(answer, str) and answer in ("a", "b") else None


@pytest.mark.parametrize("hook", _CHOOSERS)
def test_chooser_hooks_are_registrable(hook):
    assert hook in VALID_HOOKS


def test_no_plugin_means_none_without_invoking(monkeypatch):
    mgr = _install(monkeypatch, "busy_input_chooser")
    mgr._hooks.pop("busy_input_chooser")
    called = []
    monkeypatch.setattr(mgr, "invoke_hook", lambda *a, **k: called.append(1) or [])
    assert first_plugin_choice("busy_input_chooser", timeout_s=1, accept=_upper) is None
    assert called == []


def test_first_accepted_answer_wins_and_invalid_answers_are_skipped(monkeypatch):
    _install(monkeypatch, "delegation_tier_chooser",
             lambda **_: None, lambda **_: "not-a-choice", lambda **_: "b", lambda **_: "a")
    assert first_plugin_choice("delegation_tier_chooser", timeout_s=1, accept=_upper) == "B"


def test_kwargs_reach_the_callback(monkeypatch):
    seen = {}

    def cb(goal, tiers, **_):
        seen.update(goal=goal, tiers=tiers)
        return "a"

    _install(monkeypatch, "delegation_tier_chooser", cb)
    assert first_plugin_choice("delegation_tier_chooser", timeout_s=1, accept=_upper,
                               goal="g", tiers={"x": {}}) == "A"
    assert seen == {"goal": "g", "tiers": {"x": {}}}


def test_raising_callback_is_isolated(monkeypatch):
    def boom(**_):
        raise RuntimeError("plugin blew up")

    _install(monkeypatch, "delegation_tier_chooser", boom, lambda **_: "a")
    assert first_plugin_choice("delegation_tier_chooser", timeout_s=1, accept=_upper) == "A"
    _install(monkeypatch, "delegation_tier_chooser", boom)
    assert first_plugin_choice("delegation_tier_chooser", timeout_s=1, accept=_upper) is None


def test_slow_plugin_times_out_to_none(monkeypatch):
    def slow(**_):
        time.sleep(2)
        return "a"

    _install(monkeypatch, "busy_input_chooser", slow)
    started = time.monotonic()
    assert first_plugin_choice("busy_input_chooser", timeout_s=0.1, accept=_upper) is None
    assert time.monotonic() - started < 1.0
