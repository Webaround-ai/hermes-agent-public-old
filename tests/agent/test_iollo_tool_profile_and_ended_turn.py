"""Iollo fork: per-run tool profiles (``agent/tool_profile.py``) and turn-ending tools (``ends_turn``,
``agent/turn_tool_round.py``), driven through the real ``run_conversation`` loop with a fake client."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_dispatch_helpers import make_tool_result_message
from agent.tool_profile import (
    ToolProfile, lift_tool_profile_for_calls, normalize_ends_turn, normalize_skip_background_review,
    normalize_tool_profile, project_system_message,
    project_tools, set_ends_turn,
    set_tool_profile, strip_skills_block, tool_profile_report,
)
from agent.turn_tool_round import transcript_tail_ended_by_tool
from run_agent import AIAgent
from tools.registry import registry

END, END_IF, PLAIN, ESCAPE, OTHER = "t_end", "t_end_if", "t_plain", "t_escape", "t_other"

SKILLS = ("## Skills\nBefore replying, scan the skills below. If a skill matches...\n\n<available_skills>\n"
          "  misc:\n    - a: b\n</available_skills>\n\n"
          "Only proceed without loading a skill if genuinely none are relevant to the task.")
SYSTEM = "IDENTITY\n\nCONTEXT\n\n" + SKILLS + "\n\nMEMORY\n\nPLUGIN SECTIONS"


@pytest.fixture(autouse=True)
def _tools(monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)
    schema = {"description": "d", "parameters": {"type": "object", "properties": {}}}
    registry.register(name=END, toolset="iollo_test", schema={**schema, "name": END}, handler=lambda a, **k: "{}",
                      ends_turn=True)
    registry.register(name=END_IF, toolset="iollo_test", schema={**schema, "name": END_IF},
                      handler=lambda a, **k: "{}", ends_turn=lambda args, result: args.get("container") != "home")
    registry.register(name=PLAIN, toolset="iollo_test", schema={**schema, "name": PLAIN}, handler=lambda a, **k: "{}")
    registry.register(name=OTHER, toolset="iollo_test", schema={**schema, "name": OTHER}, handler=lambda a, **k: "{}")
    registry.register(name=ESCAPE, toolset="iollo_test", schema={**schema, "name": ESCAPE},
                      handler=lambda a, **k: "{}", lifts_tool_profile=True)
    yield
    for name in (END, END_IF, PLAIN, OTHER, ESCAPE):
        registry.deregister(name)


def _defs(*names):
    return [{"type": "function", "function": {"name": n, "description": n,
                                              "parameters": {"type": "object", "properties": {}}}} for n in names]


def _agent(*names):
    home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("model_tools.get_tool_definitions", return_value=_defs(*names)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = SYSTEM
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    set_ends_turn(agent, True)      # the ends_turn tests' run asked for it; see test_ends_turn_is_opt_in_per_run
    return agent


def _call(name, call_id, arguments="{}"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def _response(content="", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    reason = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=reason)], model="m", usage=None)


def _run(agent, responses, results):
    """Run one turn; ``results`` maps tool name -> result string."""
    agent.client.chat.completions.create.side_effect = responses

    def _execute(assistant_message, messages, effective_task_id, api_call_count=0):
        for tc in assistant_message.tool_calls:
            messages.append(make_tool_result_message(tc.function.name, results[tc.function.name], tc.id))

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls", side_effect=_execute),
        patch("agent.conversation_loop._restore_or_build_system_prompt"),
    ):
        return agent.run_conversation("hi", conversation_history=[{"role": "user", "content": "x"},
                                                                    {"role": "assistant", "content": "y"}])


def _sent(agent, n):
    kwargs = agent.client.chat.completions.create.call_args_list[n].kwargs
    names = [t["function"]["name"] for t in kwargs.get("tools") or []]
    system = next((m["content"] for m in kwargs["messages"] if m["role"] == "system"), "")
    return names, system


OK = json.dumps({"success": True})
BAD = json.dumps({"success": False, "error": "blocks[0]: unknown field"})


# --- ends_turn -----------------------------------------------------------------------------------------------


def test_text_with_a_succeeded_turn_ending_call_ends_the_turn_without_another_call():
    agent = _agent(END, PLAIN)
    result = _run(agent, [_response("Here are three.", [_call(END, "c1")])], {END: OK})
    assert agent.client.chat.completions.create.call_count == 1
    assert result["final_response"] == "Here are three."
    assert result["turn_exit_reason"] == "tool_ended_turn"
    assert result["completed"] is True
    # Every call keeps its tool result; the text is not repeated in a closing row.
    roles = [m["role"] for m in result["messages"]][-2:]
    assert roles == ["assistant", "tool"]
    assert sum(1 for m in result["messages"] if m.get("content") == "Here are three.") == 1
    assert transcript_tail_ended_by_tool(result["messages"]) is True


@pytest.mark.parametrize("content, calls, results", [
    ("Here.", [_call(END, "c1")], {END: BAD}),                                   # refused: the model must fix it
    ("", [_call(END, "c1")], {END: OK}),                                         # no visible text
    ("Here.", [_call(END, "c1"), _call(PLAIN, "c2")], {END: OK, PLAIN: OK}),     # a non-ending call in the round
    ("Here.", [_call(END_IF, "c1", '{"container": "home"}')], {END_IF: OK}),     # the predicate says no
])
def test_anything_else_continues_with_one_more_call(content, calls, results):
    agent = _agent(END, END_IF, PLAIN)
    result = _run(agent, [_response(content, calls), _response("Final.")], results)
    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "Final."
    assert result["turn_exit_reason"] != "tool_ended_turn"
    assert transcript_tail_ended_by_tool(result["messages"]) is False


def _responses_turn(commentary, calls):
    """One OpenAI Responses reply (gpt-6-luna writes the reply beside its tool calls as a ``phase=commentary``
    message, which the adapter files under reasoning). The fake chat client returns it as a chat completion
    carrying the raw Responses object; ``_responses_normalizer`` hands it to the real Responses transport."""
    output = ([SimpleNamespace(type="message", role="assistant", status="completed", phase="commentary", id="msg_1",
                               content=[SimpleNamespace(type="output_text", text=commentary)])] if commentary else [])
    output += [SimpleNamespace(type="function_call", status="completed", id=f"fc_{i}", call_id=f"call_{i}", name=n,
                               arguments=a) for i, (n, a) in enumerate(calls)]
    raw = SimpleNamespace(status="completed", output=output, output_text="")
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=None),
                                                    finish_reason="tool_calls")],
                           model="m", usage=None, _responses=raw)


@pytest.fixture
def _responses_normalizer(monkeypatch):
    from agent.transports.chat_completions import ChatCompletionsTransport
    from agent.transports.codex import ResponsesApiTransport
    chat_normalize = ChatCompletionsTransport.normalize_response
    responses = ResponsesApiTransport.__new__(ResponsesApiTransport)
    responses._last_issuer_kind = responses._last_issuer_model = responses._last_wire_aliases = None

    def normalize(self, response, **kwargs):
        raw = getattr(response, "_responses", None)
        if raw is None:
            return chat_normalize(self, response, **kwargs)
        normalized = ResponsesApiTransport.normalize_response(responses, raw)
        assert normalized.content == "" and normalized.finish_reason == "tool_calls"
        return normalized
    monkeypatch.setattr(ChatCompletionsTransport, "normalize_response", normalize)


def test_responses_commentary_beside_turn_ending_calls_ends_the_turn_with_that_text(_responses_normalizer):
    agent = _agent(END, PLAIN)
    deltas = []
    agent.stream_delta_callback = deltas.append
    reply = "Here are three ramen places near Chiado."
    result = _run(agent, [_responses_turn(reply, [(END, "{}")])], {END: OK})
    assert agent.client.chat.completions.create.call_count == 1
    assert result["final_response"] == reply and result["turn_exit_reason"] == "tool_ended_turn"
    rows = [m for m in result["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    # Stored once, as the visible text of the tool-call row; no flattened reasoning copy to replay again.
    assert rows[-1]["content"] == reply and not (rows[-1].get("reasoning") or "").strip()
    assert rows[-1]["codex_message_items"][0]["phase"] == "commentary"          # exact replay item kept
    assert sum(1 for m in result["messages"] if reply in str(m.get("content") or "")) == 1
    assert transcript_tail_ended_by_tool(result["messages"]) is True
    assert "".join(d for d in deltas if d).strip() == reply                   # streamed like reply text, once


@pytest.mark.parametrize("commentary, calls, results", [
    ("", [(END, "{}")], {END: OK}),                                              # no commentary: continue
    ("Here.", [(END, "{}"), (PLAIN, "{}")], {END: OK, PLAIN: OK}),               # a non-ending call: continue
    ("Here.", [(END, "{}")], {END: BAD}),                                        # refused: the model fixes it
])
def test_responses_commentary_otherwise_continues(commentary, calls, results, _responses_normalizer):
    agent = _agent(END, PLAIN)
    result = _run(agent, [_responses_turn(commentary, calls), _response("Final.")], results)
    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "Final." and result["turn_exit_reason"] != "tool_ended_turn"


def test_responses_commentary_is_left_alone_without_ends_turn(_responses_normalizer):
    agent = _agent(END)
    set_ends_turn(agent, False)
    result = _run(agent, [_responses_turn("Here.", [(END, "{}")]), _response("Final.")], {END: OK})
    assert result["final_response"] == "Final."
    row = next(m for m in result["messages"] if m.get("role") == "assistant" and m.get("tool_calls"))
    assert not row.get("content") and "Here." in (row.get("reasoning") or "")


def test_a_predicate_that_says_yes_ends_the_turn():
    agent = _agent(END_IF)
    result = _run(agent, [_response("Cards below.", [_call(END_IF, "c1", '{"container": "reply"}')])], {END_IF: OK})
    assert agent.client.chat.completions.create.call_count == 1
    assert result["final_response"] == "Cards below."


def test_a_short_reply_is_not_treated_as_a_fragment():
    agent = _agent(END)
    result = _run(agent, [_response("Done", [_call(END, "c1")])], {END: OK})
    assert result["final_response"] == "Done"


# --- tool profiles -------------------------------------------------------------------------------------------


def test_a_light_profile_projects_tools_and_drops_the_skills_index_only():
    agent = _agent(PLAIN, OTHER, END, ESCAPE)
    set_tool_profile(agent, ToolProfile("light", frozenset({PLAIN, END}), skills=False))
    result = _run(agent, [_response("Answer.")], {})
    names, system = _sent(agent, 0)
    assert names == [PLAIN, END, ESCAPE]
    assert system == "IDENTITY\n\nCONTEXT\n\nMEMORY\n\nPLUGIN SECTIONS"
    # The agent itself stays the full set (stored prompt, pins and compression see every tool).
    assert set(agent.valid_tool_names) >= {PLAIN, OTHER, END, ESCAPE}
    assert agent._cached_system_prompt == SYSTEM
    assert tool_profile_report(agent, 1) == {"name": "light", "lifted": "", "api_calls": 1}
    assert result["final_response"] == "Answer."


def test_the_profile_note_rides_on_the_escape_tool_only():
    agent = _agent(PLAIN, ESCAPE)
    set_tool_profile(agent, normalize_tool_profile({"name": "light", "tools": [PLAIN], "skills": False,
                                                    "note": "Only with every tool: Notion, the browser."}))
    _run(agent, [_response("Answer.")], {})
    tools = agent.client.chat.completions.create.call_args_list[0].kwargs["tools"]
    assert tools[1]["function"]["description"] == ESCAPE + " Only with every tool: Notion, the browser."
    assert tools[0]["function"]["description"] == PLAIN
    assert all("Notion" not in t["function"]["description"] for t in agent.tools)


def test_without_a_profile_the_escape_tool_is_never_sent():
    agent = _agent(PLAIN, ESCAPE)
    set_tool_profile(agent, None)
    _run(agent, [_response("Answer.")], {})
    names, system = _sent(agent, 0)
    assert names == [PLAIN]
    assert system == SYSTEM
    assert tool_profile_report(agent) is None


@pytest.mark.parametrize("called, reason", [(ESCAPE, "escape"), (OTHER, "tool:" + OTHER)])
def test_a_call_outside_the_profile_lifts_it_for_the_rest_of_the_turn(called, reason):
    agent = _agent(PLAIN, OTHER, ESCAPE)
    set_tool_profile(agent, ToolProfile("light", frozenset({PLAIN}), skills=False))
    result = _run(agent, [_response("", [_call(called, "c1")]), _response("Done with all tools.")],
                  {called: OK})
    first, _ = _sent(agent, 0)
    second, system = _sent(agent, 1)
    assert first == [PLAIN, ESCAPE]
    assert second == [PLAIN, OTHER]
    assert system == SYSTEM
    assert tool_profile_report(agent)["lifted"] == reason
    assert result["final_response"] == "Done with all tools."


# --- connector globs and read tools --------------------------------------------------------------------------

LIST, GET, CREATE, LYING, NOTION = ("mcp__linear__list_issues", "mcp__linear__get_issue", "mcp__linear__create_issue",
                                    "mcp__linear__updateIssue", "mcp__notion__search")


@pytest.fixture
def _mcp(monkeypatch):
    """Discovery records as the MCP client leaves them: provenance per registry name and readOnlyHint per raw name
    (Notion published no annotations)."""
    import tools.mcp_tool as core
    from tools.mcp_tool_scope import _resolve_server_key
    for name, server in ((LIST, "linear"), (GET, "linear"), (CREATE, "linear"), (LYING, "linear"),
                         (NOTION, "notion")):
        monkeypatch.setitem(core._mcp_tool_server_names, name, server)
    monkeypatch.setitem(core._tool_read_only_hints, _resolve_server_key("linear"),
                        {"list_issues": True, "get_issue": True, "create_issue": False, "updateIssue": True})


def _connector_agent():
    agent = _agent(PLAIN, LIST, CREATE, GET, LYING, NOTION, ESCAPE)
    agent._skip_mcp_refresh = True
    return agent


def test_read_tools_send_only_annotated_read_only_tools_in_session_order(_mcp):
    agent = _connector_agent()
    set_tool_profile(agent, normalize_tool_profile({"name": "light", "tools": [PLAIN], "skills": False,
                                                    "read_tools": ["mcp__linear__*", "mcp__notion__*"]}))
    _run(agent, [_response("Answer.")], {})
    names, _ = _sent(agent, 0)
    # No create (readOnlyHint false), no updateIssue (annotated read-only but a write verb), no Notion (no annotation).
    assert names == [PLAIN, LIST, GET, ESCAPE]
    report = tool_profile_report(agent, 1)["read_tools"]
    assert report["status"] == "ok" and report["tools"] == 2 and report["chars"] > 0
    assert report["patterns"] == {"mcp__linear__*": 2, "mcp__notion__*": 0}


def test_a_read_tool_call_keeps_the_profile_and_a_write_call_lifts_it(_mcp):
    agent = _connector_agent()
    set_tool_profile(agent, ToolProfile("light", frozenset({PLAIN}), skills=False, read_patterns=("mcp__linear__*",)))
    _run(agent, [_response("", [_call(LIST, "c1")]), _response("", [_call(CREATE, "c2")]), _response("Done.")],
         {LIST: OK, CREATE: OK})
    assert _sent(agent, 1)[0] == [PLAIN, LIST, GET, ESCAPE]
    assert _sent(agent, 2)[0] == [PLAIN, LIST, CREATE, GET, LYING, NOTION]
    assert tool_profile_report(agent)["lifted"] == "tool:" + CREATE


def test_read_tools_over_budget_send_none_of_them(_mcp, monkeypatch):
    monkeypatch.setattr("agent.tool_profile.MAX_READ_TOOLS", 1)
    agent = _connector_agent()
    set_tool_profile(agent, ToolProfile("light", frozenset({PLAIN}), skills=False, read_patterns=("mcp__linear__*",)))
    _run(agent, [_response("Answer.")], {})
    assert _sent(agent, 0)[0] == [PLAIN, ESCAPE]
    assert tool_profile_report(agent)["read_tools"]["status"] == "over_budget"


def test_read_tools_never_include_a_tool_the_mcp_client_did_not_register(monkeypatch):
    """Without the MCP client's records (or with it not imported) nothing counts as read-only: fail closed."""
    import tools.mcp_tool as core
    monkeypatch.setattr(core, "_mcp_tool_server_names", {})
    agent = _connector_agent()
    set_tool_profile(agent, ToolProfile("light", frozenset({PLAIN}), read_patterns=("mcp__linear__*",)))
    _run(agent, [_response("Answer.")], {})
    assert _sent(agent, 0)[0] == [PLAIN, ESCAPE]
    assert tool_profile_report(agent)["read_tools"]["status"] == "none"


def test_globs_in_tools_resolve_per_request_and_unknown_ones_are_ignored():
    agent = _agent(PLAIN, OTHER, ESCAPE)
    # An unknown exact name (a relay tool an older box image lacks, e.g. search_answer) is ignored the same way.
    set_tool_profile(agent, normalize_tool_profile({"name": "light", "tools": ["t_pl*", "zzzz*", "search_answer"]}))
    _run(agent, [_response("", [_call(PLAIN, "c1")]), _response("Answer.")], {PLAIN: OK})
    assert _sent(agent, 0)[0] == [PLAIN, ESCAPE]
    assert _sent(agent, 1)[0] == [PLAIN, ESCAPE]
    assert "read_tools" not in tool_profile_report(agent)


def test_normalize_keeps_globs_and_read_tools_apart():
    profile = normalize_tool_profile({"name": "light", "tools": ["web_search", "mcp__x__*", "mcp__x__*"],
                                      "read_tools": ["mcp__y__*", "mcp__z__get"]})
    assert profile == ToolProfile("light", frozenset({"web_search"}), True, "", ("mcp__x__*",),
                                  ("mcp__y__*", "mcp__z__get"))
    for bad in ({"tools": ["*"]}, {"tools": ["mc*"]}, {"tools": ["x"], "read_tools": ["a?bc*"]},
                {"tools": ["x"], "read_tools": "mcp__y__*"}, {"tools": ["x"], "read_tools": ["mcp__y__*"] * 21},
                {"name": "full", "read_tools": ["mcp__y__*"]}):
        with pytest.raises(ValueError):
            normalize_tool_profile(bad)


def test_normalize_skip_background_review():
    assert normalize_skip_background_review(None) is False
    assert normalize_skip_background_review(True) is True
    with pytest.raises(ValueError):
        normalize_skip_background_review("yes")


# --- helpers -------------------------------------------------------------------------------------------------


def test_strip_skills_block_keeps_every_other_byte():
    assert strip_skills_block(SYSTEM) == "IDENTITY\n\nCONTEXT\n\nMEMORY\n\nPLUGIN SECTIONS"
    assert strip_skills_block("no skills here") == "no skills here"
    hidden = SKILLS + ("\n(Categories marked [names only] are outside the current coding context, so their "
                       "descriptions are omitted — the skills work normally and load with skill_view(name) as usual.)")
    assert strip_skills_block("A\n\n" + hidden + "\n\nB") == "A\n\nB"
    # An unterminated block is left alone rather than cutting the rest of the prompt.
    assert strip_skills_block("A\n\n## Skills\nBefore replying, scan the skills below. ...") == (
        "A\n\n## Skills\nBefore replying, scan the skills below. ...")


def test_project_system_message_never_mutates_the_input():
    agent = SimpleNamespace(_tool_profile=ToolProfile("light", frozenset({"a"}), skills=False))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    projected = project_system_message(agent, messages)
    assert messages[0]["content"] == SYSTEM
    assert projected[0]["content"].startswith("IDENTITY") and "## Skills" not in projected[0]["content"]
    blocks = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    assert "## Skills" not in project_system_message(agent, blocks)[0]["content"][0]["text"]
    keep = SimpleNamespace(_tool_profile=ToolProfile("light", frozenset({"a"}), skills=True))
    assert project_system_message(keep, messages) is messages


def test_project_tools_returns_the_same_list_when_nothing_is_removed():
    agent = SimpleNamespace(_tool_profile=None)
    tools = _defs(PLAIN, OTHER)
    assert project_tools(agent, tools) is tools


def test_explicit_profile_materializes_scoped_deferred_tool_and_dispatches_directly(monkeypatch):
    """A lookup profile can name a deferred plugin without adding discovery bridges."""
    import model_tools
    from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig, assemble_tool_defs

    name = "iollo_profile_search_answer"
    hidden = "iollo_profile_hidden_answer"
    toolset = "mcp-iollo-profile-test"
    calls = []
    schema = {"name": name, "description": "Read the owner's scoped profile.",
              "parameters": {"type": "object", "properties": {
                  "query": {"type": "string"}}, "required": ["query"]}}
    hidden_schema = {"name": hidden, "description": "Out of scope.",
                     "parameters": {"type": "object", "properties": {}}}
    registry.register(name=name, toolset=toolset, schema=schema,
                      handler=lambda args: calls.append(args["query"]) or json.dumps({"answer": "found"}))
    registry.register(name=hidden, toolset="mcp-iollo-other-test", schema=hidden_schema,
                      handler=lambda args: "should not dispatch")
    try:
        raw = model_tools.get_tool_definitions(
            enabled_toolsets=[toolset], quiet_mode=True, skip_tool_search_assembly=True)
        assert [item["function"]["name"] for item in raw] == [name]
        assembled = assemble_tool_defs(
            raw, context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on", "listing": "off"}),
        )
        assert assembled.activated
        assert BRIDGE_TOOL_NAMES <= {item["function"]["name"] for item in assembled.tool_defs}
        assert name not in {item["function"]["name"] for item in assembled.tool_defs}

        agent = SimpleNamespace(
            tools=assembled.tool_defs,
            valid_tool_names={item["function"]["name"] for item in assembled.tool_defs},
            enabled_toolsets=[toolset], disabled_toolsets=None,
        )
        stored_pin = list(agent.tools)
        set_tool_profile(agent, ToolProfile("lookup", frozenset({name})))
        projected = project_tools(agent, agent.tools)
        assert [item["function"]["name"] for item in projected] == [name]
        assert agent.tools == stored_pin
        assert name in agent.valid_tool_names
        assert hidden not in agent.valid_tool_names

        result = model_tools.handle_function_call(
            name, {"query": "remembered detail"}, enabled_tools=list(agent.valid_tool_names),
            enabled_toolsets=agent.enabled_toolsets, disabled_toolsets=agent.disabled_toolsets,
        )
        assert json.loads(result) == {"answer": "found"}
        assert calls == ["remembered detail"]

        # A later unprofiled request restores the normal assembled pin and does
        # not keep the request-local direct-dispatch grant.
        agent._tool_profile = None
        project_tools(agent, agent.tools)
        assert name not in agent.valid_tool_names
        assert agent.tools == stored_pin
    finally:
        registry.deregister(name)
        registry.deregister(hidden)


@pytest.mark.parametrize("read_cap, expected_reads, status", [
    (12, {LIST, GET}, "ok"),
    (1, set(), "over_budget"),
])
def test_deferred_read_profile_keeps_annotations_write_exclusions_and_budget(
    _mcp, monkeypatch, read_cap, expected_reads, status,
):
    import tools.mcp_tool as core
    from tools.mcp_tool_scope import _resolve_server_key
    from tools.tool_search import ToolSearchConfig, assemble_tool_defs

    toolset = "mcp-linear"
    registered = (LIST, GET, CREATE, LYING)
    schema = lambda name: {"name": name, "description": name,
                           "parameters": {"type": "object", "properties": {}}}
    for name in registered:
        registry.register(name=name, toolset=toolset, schema=schema(name), handler=lambda args: "{}")
    try:
        server_key = _resolve_server_key("linear")
        for name in registered:
            monkeypatch.setitem(core._mcp_tool_server_names, name, "linear")
        monkeypatch.setitem(core._tool_read_only_hints, server_key,
                            {"list_issues": True, "get_issue": True, "create_issue": False,
                             "updateIssue": True})
        import model_tools
        raw = model_tools.get_tool_definitions(
            enabled_toolsets=[toolset, "iollo_test"], quiet_mode=True, skip_tool_search_assembly=True)
        assembled = assemble_tool_defs(
            raw, context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on", "listing": "off"}),
        )
        assert assembled.activated
        agent = SimpleNamespace(
            tools=assembled.tool_defs,
            valid_tool_names={item["function"]["name"] for item in assembled.tool_defs},
            enabled_toolsets=[toolset, "iollo_test"], disabled_toolsets=None,
        )
        if read_cap != 12:
            monkeypatch.setattr("agent.tool_profile.MAX_READ_TOOLS", read_cap)
        set_tool_profile(agent, ToolProfile(
            "light", frozenset({PLAIN}), read_patterns=("mcp__linear__*",)))
        projected = project_tools(agent, agent.tools)
        names = {item["function"]["name"] for item in projected}
        assert PLAIN in names and ESCAPE in names
        assert names & {LIST, GET, CREATE, LYING} == expected_reads
        assert CREATE not in names and LYING not in names
        assert agent._tool_profile_reads["status"] == status
    finally:
        for name in registered:
            registry.deregister(name)


def test_deferred_profile_catalog_failure_is_a_visible_request_error(monkeypatch):
    agent = SimpleNamespace(
        tools=_defs("tool_search"), valid_tool_names={"tool_search"},
        enabled_toolsets=["mcp-missing-catalog"], disabled_toolsets=None,
    )
    set_tool_profile(agent, ToolProfile("lookup", frozenset({"mcp_missing_lookup"})))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: (_ for _ in ()).throw(OSError("catalog down")))
    with pytest.raises(RuntimeError, match="Unable to resolve this run's scoped tool profile"):
        project_tools(agent, agent.tools)


def test_profile_catalog_keeps_oneshot_and_side_agent_drops(monkeypatch):
    import model_tools

    agent = SimpleNamespace(
        tools=_defs("tool_search"), valid_tool_names={"tool_search"}, side_agent=True,
        enabled_toolsets=["test-scope"], disabled_toolsets=None,
    )
    set_tool_profile(agent, ToolProfile("limited", frozenset({"skill_manage", "manage_connections"})))
    monkeypatch.setattr("agent.oneshot_footprint.is_single_query_session", lambda: True)
    monkeypatch.setattr(model_tools, "get_tool_definitions",
                        lambda **kwargs: _defs("skill_manage", "manage_connections"))

    assert project_tools(agent, agent.tools) == []
    assert agent.valid_tool_names == {"tool_search"}


def test_lift_ignores_unknown_names_and_profile_tools():
    agent = SimpleNamespace(_tool_profile=ToolProfile("light", frozenset({PLAIN})), valid_tool_names={PLAIN, OTHER})
    assert lift_tool_profile_for_calls(agent, [_call(PLAIN, "c"), _call("nope", "d")]) is False
    assert agent._tool_profile is not None


@pytest.mark.parametrize("value, expected", [
    (None, None),
    ({"name": "light", "tools": ["web_search"]}, ToolProfile("light", frozenset({"web_search"}), True)),
    ({"name": "full"}, ToolProfile("full", None, True)),
])
def test_normalize_tool_profile(value, expected):
    assert normalize_tool_profile(value) == expected


def test_a_named_profile_without_tools_projects_nothing_but_reports():
    agent = _agent(PLAIN, OTHER, ESCAPE)
    set_tool_profile(agent, ToolProfile("full", None))
    _run(agent, [_response("Answer.")], {})
    names, system = _sent(agent, 0)
    assert names == [PLAIN, OTHER]
    assert system == SYSTEM
    assert tool_profile_report(agent, 1) == {"name": "full", "lifted": "", "api_calls": 1}


def test_ends_turn_is_opt_in_per_run():
    """Off unless the run asks (/v1/runs ``ends_turn: true``): flows write task records after the reply's cards."""
    agent = _agent(END)
    set_ends_turn(agent, False)
    result = _run(agent, [_response("Here.", [_call(END, "c1")]), _response("Final.")], {END: OK})
    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "Final."
    assert normalize_ends_turn(None) is False and normalize_ends_turn(True) is True
    with pytest.raises(ValueError):
        normalize_ends_turn("yes")


def test_a_stop_during_the_round_is_not_turned_into_a_completed_reply():
    agent = _agent(END)
    agent.client.chat.completions.create.side_effect = [_response("Here.", [_call(END, "c1")]), _response("x")]

    def _execute(assistant_message, messages, effective_task_id, api_call_count=0):
        messages.append(make_tool_result_message(END, OK, "c1"))
        agent._interrupt_requested = True

    with (
        patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls", side_effect=_execute),
        patch("agent.conversation_loop._restore_or_build_system_prompt"),
    ):
        result = agent.run_conversation("hi")
    assert result["turn_exit_reason"] != "tool_ended_turn"
    assert result["interrupted"] is True


def test_a_tool_ended_turn_on_the_last_allowed_call_is_completed():
    agent = _agent(END)
    agent.max_iterations = 1
    result = _run(agent, [_response("Here are three.", [_call(END, "c1")])], {END: OK})
    assert result["turn_exit_reason"] == "tool_ended_turn"
    assert result["completed"] is True and result["final_response"] == "Here are three."


def test_ascii_recovery_never_rewrites_the_canonical_tool_schemas():
    from agent.message_sanitization import sanitize_outbound_kwargs
    agent = _agent(PLAIN, ESCAPE)
    agent.tools[0]["function"]["description"] = "caf\u00e9"
    canonical = agent.tools[0]["function"]["description"]
    set_tool_profile(agent, None)
    projected = project_tools(agent, agent.tools)
    assert projected is not agent.tools and projected[0] is agent.tools[0]
    agent._force_ascii_payload = True
    kwargs = {"tools": projected, "messages": []}
    sanitize_outbound_kwargs(agent, kwargs)
    assert agent.tools[0]["function"]["description"] == canonical


def test_explicit_publication_text_closes_without_another_model_call(monkeypatch):
    monkeypatch.setattr(registry.get_entry(END_IF), 'ends_turn', lambda args, result: args['reply_text'])
    agent = _agent(END_IF)
    result = _run(agent, [_response('', [_call(END_IF, 'c1', '{"reply_text":"Here is the explanation."}')])], {END_IF: OK})
    assert agent.client.chat.completions.create.call_count == 1
    assert result['final_response'] == 'Here is the explanation.'
    assert result['messages'][-1]['role'] == 'assistant'
    assert result['messages'][-1]['content'] == result['final_response']
    row = next(m for m in result['messages'] if m.get('tool_calls'))
    assert not row.get('content')  # the persisted call row is not rewritten


@pytest.mark.parametrize('failure,mixed', [(True, False), (False, True)])
def test_publication_text_does_not_hide_failed_or_mixed_rounds(monkeypatch, failure, mixed):
    monkeypatch.setattr(registry.get_entry(END_IF), 'ends_turn', lambda args, result: args['reply_text'])
    agent = _agent(END_IF, PLAIN)
    calls = [_call(END_IF, 'c1', '{"reply_text":"Premature success"}')]
    if mixed:
        calls.append(_call(PLAIN, 'c2'))
    result = _run(agent, [_response('', calls), _response('Recovered.')],
                  {END_IF: BAD if failure else OK, PLAIN: OK})
    assert agent.client.chat.completions.create.call_count == 2
    assert result['final_response'] == 'Recovered.'
