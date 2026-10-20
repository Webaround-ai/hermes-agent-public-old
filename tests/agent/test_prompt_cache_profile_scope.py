"""Iollo fork: ``prompt_caching.key_scope: profile`` shares one prompt_cache_key scope across conversations."""
from types import SimpleNamespace

import agent.chat_completion_helpers as helpers


def _agent(session_id):
    return SimpleNamespace(session_id=session_id, _session_db=None)


def test_profile_scope_is_one_scope_for_every_conversation(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"prompt_caching": {"key_scope": "profile"}})
    assert helpers._prompt_cache_scope_for_agent(_agent("run_a")) == "profile"
    assert helpers._prompt_cache_scope_for_agent(_agent("run_b")) == "profile"


def test_the_default_keeps_the_conversation_scope(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"prompt_caching": {"cache_ttl": "1h"}})
    assert helpers._prompt_cache_scope_for_agent(_agent("run_a")) == "run_a"
