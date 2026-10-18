"""One tool-calling round of the conversation turn loop: validate/cap/dedupe the model's
tool calls, persist the tool-call turn BEFORE any side effect, execute the tools, honour
guardrail halts / persistence failures, then compress after tool results. Nothing here
imports ``agent.conversation_loop`` at module level (cycle) — loop-internal helpers resolve
lazily.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Optional, Tuple

from agent.message_metadata import append_message
from agent.message_sanitization import coalesce_tool_call_id
from agent.turn_preflight import compress_after_tool_results
from agent.turn_tool_validation import validate_tool_calls

logger = logging.getLogger("agent.conversation_loop")

# Post-response housekeeping tools: a round made only of these mutes tool progress.
_HOUSEKEEPING_TOOLS = frozenset({"memory", "todo_list", "skill_manage", "session_search"})


@dataclass
class ToolRoundVerdict:
    """``action``: ``"continue"`` (tools ran, next API call), ``"break"`` (turn ends:
    persistence failure, guardrail halt, post-tool compression end) or ``"return"``
    (``result`` is the turn's result dict). The other fields are the loop locals the round
    rebinds."""

    action: str
    messages: Any
    conversation_history: Any
    active_system_prompt: Any
    compression_attempts: Any
    final_response: Any
    failed: Any
    _turn_exit_reason: Any
    truncated_tool_call_retries: Any
    current_turn_user_idx: Any
    result: Optional[Dict[str, Any]] = None


def run_tool_round(
    agent: Any, *, assistant_message: Any, finish_reason: Any, messages: Any,
    conversation_history: Any, api_call_count: Any, effective_task_id: Any, user_message: Any,
    system_message: Any, active_system_prompt: Any, compression_attempts: Any,
    max_compression_attempts: Any, final_response: Any, failed: Any, _turn_exit_reason: Any,
    truncated_tool_call_retries: Any, current_turn_user_idx: Any,
) -> ToolRoundVerdict:
    """Execute one tool round in the exact original order. Persist-before-execute is a
    durability invariant: resume must see the executed block if a destructive tool restarts
    Hermes; a failed canonical append ends the turn rather than running tools from
    process-only state."""
    from agent.conversation_loop import _invalid_tool_name_error_content

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ToolRoundVerdict:
        return ToolRoundVerdict(
            action=action, messages=messages, conversation_history=conversation_history,
            active_system_prompt=active_system_prompt, compression_attempts=compression_attempts,
            final_response=final_response, failed=failed, _turn_exit_reason=_turn_exit_reason,
            truncated_tool_call_retries=truncated_tool_call_retries,
            current_turn_user_idx=current_turn_user_idx, result=result,
        )

    if not agent.quiet_mode:
        agent._vprint(f"{agent.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")

    if agent.verbose_logging:
        for tc in assistant_message.tool_calls:
            raw_args = tc.function.arguments
            args_preview = raw_args[:200] if isinstance(raw_args, str) else repr(raw_args)[:200]
            logging.debug("Tool call: %s with args: %s...", tc.function.name, args_preview)

    _tvv = validate_tool_calls(
        agent, assistant_message, finish_reason, messages=messages,
        conversation_history=conversation_history, api_call_count=api_call_count,
        effective_task_id=effective_task_id,
    )
    if _tvv.action == "return":
        return _verdict("return", _tvv.result)
    if _tvv.action == "continue":
        return _verdict("continue")

    # Iollo fork: a call outside this run's tool profile (or to its escape tool) lifts the profile, so the
    # call runs as usual and the next request carries every tool (agent/tool_profile.py).
    from agent.tool_profile import lift_tool_profile_for_calls
    lift_tool_profile_for_calls(agent, assistant_message.tool_calls)

    # Post-call guardrails.
    assistant_message.tool_calls = agent._deduplicate_tool_calls(
        agent._cap_delegate_task_calls(assistant_message.tool_calls)
    )

    # Mixed batch: the assistant message keeps EVERY emitted call (each tool_call needs a
    # matching result) while only valid ones dispatch.
    _invalid_batch_calls = [
        tc for tc in assistant_message.tool_calls if tc.function.name not in agent.valid_tool_names
    ] if _tvv.mixed_invalid_batch else []

    assistant_msg, duplicate_previous_interim = stage_tool_call_message(
        agent, assistant_message=assistant_message, finish_reason=finish_reason, messages=messages
    )
    append_message(messages, assistant_msg)

    # Mixed batch: error-result invalid calls and drop them from execution.
    if _invalid_batch_calls:
        for tc in _invalid_batch_calls:
            append_message(messages, {
                "role": "tool",
                "name": tc.function.name,
                "tool_call_id": coalesce_tool_call_id(tc),
                "content": _invalid_tool_name_error_content(
                    tc.function.name, agent.valid_tool_names
                ),
            })
        assistant_message.tool_calls = [
            tc for tc in assistant_message.tool_calls if tc.function.name in agent.valid_tool_names
        ]

    # Persist the tool-call turn before any tool side effects so resume sees the executed
    # block if a destructive tool restarts Hermes.
    try:
        _tool_turn_persisted = agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception as exc:
        _tool_turn_persisted = False
        from hermes_state import classify_persistence_error
        agent._last_persistence_error_cause = classify_persistence_error(exc)
        logger.warning(
            "Incremental tool-call persistence failed before execution "
            "(session=%s): %s",
            agent.session_id or "none",
            exc,
        )

    if _tool_turn_persisted is False:
        # Canonical append failed: never project the row or run tools from process-only
        # state; break rather than retry. No recorded cause means genuinely unknown.
        if getattr(agent, "_last_persistence_error_cause", None) is None:
            agent._last_persistence_error_cause = "unknown"
        _turn_exit_reason = "session_persistence_failed"
        final_response = ""
        failed = True
        return _verdict("break")

    # A UI must never observe an assistant/tool-call row that is only an in-memory
    # projection: emit interim commentary after the DB append.
    if not duplicate_previous_interim:
        agent._emit_interim_assistant_message(assistant_msg)

    # Flush open streaming boxes before tools so early content doesn't wrap tool feed
    # lines. Display callback only — TTS (_stream_callback) must NOT receive None (EOS).
    if agent.stream_delta_callback:
        with suppress(Exception):
            agent.stream_delta_callback(None)

    agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

    if getattr(agent, "_incremental_persistence_failed", False):
        # Tool result could not be made canonical: never send the in-memory result to
        # the model or project later events from this turn.
        _turn_exit_reason = "session_persistence_failed"
        final_response = ""
        failed = True
        return _verdict("break")

    if agent._tool_guardrail_halt_decision is not None:
        decision = agent._tool_guardrail_halt_decision
        _turn_exit_reason = "guardrail_halt"
        final_response = agent._toolguard_controlled_halt_response(decision)
        agent._emit_diagnostic_status(f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}")
        append_message(messages, {"role": "assistant", "content": final_response})
        # Emit the halt so it isn't mistaken for a crash; the stream callback is still
        # alive, so SSE/TUI clients see the explanation.
        if final_response:
            agent._safe_print(f"\n{final_response}\n")
            if agent.stream_delta_callback:
                with suppress(Exception):
                    agent.stream_delta_callback(final_response)
                    agent.stream_delta_callback(None)
        return _verdict("break")

    # Iollo fork: a round made only of turn-ending tools that all succeeded, whose message carried visible
    # text, ends the turn with that text instead of one more model call (``tool_ended_turn``).
    ended_text = tool_ended_turn_text(agent, assistant_message, messages)
    if ended_text:
        if getattr(agent, "_tool_ended_turn_payload", False):
            # The reply came from the successful tool's explicit publication payload. Append a
            # closing assistant row; never rewrite the already-persisted tool-call row.
            append_message(messages, {"role": "assistant", "content": ended_text})
            try:
                persisted = agent._flush_messages_to_session_db(messages, conversation_history)
            except Exception as exc:
                from hermes_state import classify_persistence_error
                agent._last_persistence_error_cause = classify_persistence_error(exc)
                persisted = False
                logger.warning("Final publication persistence failed: %s", type(exc).__name__)
            if persisted is False:
                _turn_exit_reason = "session_persistence_failed"
                final_response, failed = "", True
                return _verdict("break")
        _turn_exit_reason = "tool_ended_turn"
        final_response = ended_text
        agent._turn_ended_by_tool = True
        if (getattr(agent, "_tool_ended_turn_commentary", None) == ended_text
                or getattr(agent, "_tool_ended_turn_payload", False)):
            # Promoted Responses commentary only went out as interim commentary: stream it once as reply text,
            # like a Claude reply's own deltas (the run's output carries it either way).
            with suppress(Exception):
                agent._fire_stream_delta(ended_text)
            if agent.stream_delta_callback:
                with suppress(Exception):
                    agent.stream_delta_callback(None)
        # The text already went out as this message's stream deltas / interim commentary.
        agent._response_was_previewed = True
        agent._session_messages = messages
        return _verdict("break")

    # Reset per-turn retry counters so one truncation can't poison the turn.
    truncated_tool_call_retries = 0
    # Defer the paragraph break: _fire_stream_delta() prepends one "\n\n" when real
    # text arrives, so tool iterations don't stack blank lines.
    agent._stream_needs_break = True
    # Refund the iteration when the ONLY tool was execute_code (programmatic tool
    # calling) — cheap RPC-style calls shouldn't eat the budget.
    if {tc.function.name for tc in assistant_message.tool_calls} == {"execute_code"}:
        agent.iteration_budget.refund()

    _ptc = compress_after_tool_results(
        agent, messages=messages, system_message=system_message, user_message=user_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts, effective_task_id=effective_task_id,
        final_response=final_response, turn_exit_reason=_turn_exit_reason,
        current_turn_user_idx=current_turn_user_idx,
    )
    messages = _ptc.messages
    active_system_prompt = _ptc.active_system_prompt
    conversation_history = _ptc.conversation_history
    compression_attempts = _ptc.compression_attempts
    final_response = _ptc.final_response
    _turn_exit_reason = _ptc.turn_exit_reason
    current_turn_user_idx = _ptc.current_turn_user_idx
    if _ptc.end_turn:
        return _verdict("break")

    # Save session log incrementally (so progress is visible even if interrupted)
    agent._session_messages = messages
    # Touch activity so slow post-tool work plus a slow follow-up API call can't exceed
    # the gateway inactivity timeout (HERMES_AGENT_TIMEOUT).
    # Touch activity before continuing so the gateway's inactivity monitor never sees a stale timestamp
    # between tool completion and the start of the next API call. Without this, a tool-call result (which
    # takes ~0s to process) followed by slow post-tool processing (compression, persist) and a slow
    # follow-up API call can exceed the gateway inactivity timeout (HERMES_AGENT_TIMEOUT, default 1800s) and
    # the gateway kills the session before the next activity touch fires (#69559, #69131).
    agent._touch_activity(f"tool results posted, continuing iteration #{api_call_count}")
    return _verdict("continue")


def tool_ended_turn_text(agent: Any, assistant_message: Any, messages: Any) -> Optional[str]:
    """The visible text that ends the turn after this round, or ``None`` to continue as usual.

    Every call must name a tool registered ``ends_turn`` (``True``, or a ``predicate(args, result)`` that
    returns True or explicit final reply text), have a recorded result that is not a failure (``agent.display._detect_tool_failure``),
    and the assistant message must carry visible text. Any doubt continues the loop: a failed or refused
    call reaches the model, which can fix it."""
    agent._tool_ended_turn_payload = False
    # Execution transfer is native control flow, independent of reply-tool opt-in.
    # Its trusted predicate ran on the canonical executed tool after result persistence;
    # remaining sibling calls have already been recorded as not started.
    handoff = getattr(agent, '_tool_execution_handoff', None)
    if handoff and getattr(agent, '_interrupt_requested', False) is not True:
        agent._tool_ended_turn_payload = True
        return handoff
    # Opt-in per run (``ends_turn: true`` on /v1/runs): skills and flows that write records after the reply's cards
    # (task_update after show_widget) must keep their next round on every other run.
    if getattr(agent, "_ends_turn_enabled", False) is not True:
        return None
    # A stop requested during the round wins: the next iteration reports the turn cancelled.
    if getattr(agent, "_interrupt_requested", False) is True:
        return None
    calls = list(getattr(assistant_message, "tool_calls", None) or [])
    if not calls:
        return None
    from agent.conversation_loop import _STALE_MARKER_RE
    raw = assistant_message.content if isinstance(getattr(assistant_message, "content", None), str) else ""
    text = (agent._strip_think_blocks(raw).strip() if raw and
            not _STALE_MARKER_RE.fullmatch(raw.strip()) and agent._has_content_after_think_block(raw) else "")
    published = []
    from agent.display import _detect_tool_failure
    from tools.registry import registry
    ids = {coalesce_tool_call_id(tc) for tc in calls}
    results = {}
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            break
        if msg.get("tool_call_id") in ids:
            results.setdefault(msg.get("tool_call_id"), msg.get("content"))
    for tc in calls:
        name = getattr(tc.function, "name", "")
        entry = registry.get_entry(name) if name else None
        rule = getattr(entry, "ends_turn", False) if entry is not None else False
        if not rule:
            return None
        result = results.get(coalesce_tool_call_id(tc))
        if result is None or _detect_tool_failure(name, result)[0]:
            return None
        if callable(rule):
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    return None
            try:
                decision = rule(args if isinstance(args, dict) else {}, result)
                if isinstance(decision, str):
                    if not decision.strip() or len(decision) > 8000:
                        return None
                    published.append(decision)
                elif decision is not True:
                    return None
            except Exception:  # noqa: BLE001 - a broken predicate never ends a turn
                logger.debug("ends_turn predicate for %s raised", name, exc_info=True)
                return None
    if published:
        # Multiple distinct final answers are ambiguous; let the model reconcile them.
        if len(set(published)) != 1:
            return None
        raw = published[0]
        if _STALE_MARKER_RE.fullmatch(raw.strip()) or not agent._has_content_after_think_block(raw):
            return None
        text = agent._strip_think_blocks(raw).strip()
        agent._tool_ended_turn_payload = bool(text)
    return text or None


def _all_turn_ending(calls: Any) -> bool:
    """Every call names a tool registered ``ends_turn`` (a predicate counts; it is judged after the round)."""
    from tools.registry import registry
    for tc in calls:
        name = getattr(getattr(tc, "function", None), "name", "")
        entry = registry.get_entry(name) if name else None
        if entry is None or not getattr(entry, "ends_turn", False):
            return False
    return bool(calls)


def _promote_commentary_reply(agent: Any, assistant_message: Any, assistant_msg: Dict[str, Any]) -> str:
    """Iollo fork: on the OpenAI Responses path the text the model writes beside its tool calls arrives as
    ``phase=commentary`` message items, which the adapter files under reasoning, so a turn-ending round
    (``show_widget``/``ask_owner`` with the reply beside them) had no visible text and never ended. On a run that
    asked for ``ends_turn``, when the message has no content and every call names a turn-ending tool, that
    commentary becomes the row's visible content (and leaves its flattened reasoning copy) before the row is
    persisted. The exact ``codex_message_items`` stay, so the Responses replay is byte-identical (content is not
    replayed beside them). Returns the promoted text ("" when nothing changed); Claude paths never carry
    commentary items."""
    agent._tool_ended_turn_commentary = None
    if getattr(agent, "_ends_turn_enabled", False) is not True or getattr(agent, "_interrupt_requested", False) is True:
        return ""
    content = assistant_message.content if isinstance(getattr(assistant_message, "content", None), str) else ""
    if content.strip() or not _all_turn_ending(list(getattr(assistant_message, "tool_calls", None) or [])):
        return ""
    from agent.history_commentary import _commentary_items, _without_flattened_commentary, visible_commentary
    raw = _commentary_items({"codex_message_items": getattr(assistant_message, "codex_message_items", None)})
    parts = [p for text in raw if (p := visible_commentary(text, strip_thinking=agent._strip_think_blocks))]
    text = "\n\n".join(parts).strip()
    if not text:
        return ""
    assistant_message.content = text
    assistant_msg["content"] = text
    for holder, get, put in ((assistant_msg, assistant_msg.get, assistant_msg.__setitem__),
                             (assistant_message, lambda k: getattr(assistant_message, k, None),
                              lambda k, v: setattr(assistant_message, k, v))):
        reasoning = get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            put("reasoning", _without_flattened_commentary(reasoning, raw).strip() or None)
    agent._tool_ended_turn_commentary = text
    return text


def stage_tool_call_message(
    agent: Any, *, assistant_message: Any, finish_reason: Any, messages: Any
) -> Tuple[Dict[str, Any], bool]:
    """Build the assistant tool-call row and update the per-turn fallback/mute state.

    Drops a bare bracketed marker beside a call (#78148), classifies housekeeping-only
    rounds, keeps visible content as the empty-follow-up fallback, pops thinking-only
    prefills (resetting their counters), re-arms the post-tool nudge and the
    dropped-tool-call stall budget. Returns ``(assistant_msg, duplicate_previous_interim)``;
    the flag suppresses re-emitting interim commentary the previous ``incomplete`` row
    already showed."""
    from agent.conversation_loop import _STALE_MARKER_RE

    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
    _promote_commentary_reply(agent, assistant_message, assistant_msg)

    turn_content = assistant_message.content or ""

    # A bare bracketed token (e.g. ``[memory]``) beside a function call is protocol
    # scaffolding; persisting it lets the post-tool fallback replay it forever (#78148).
    if assistant_message.tool_calls and _STALE_MARKER_RE.fullmatch(turn_content.strip()):
        logger.warning(
            "Discarding bare tool-call marker from assistant content: %s", turn_content
        )
        turn_content = ""
        assistant_msg["content"] = ""

    # Classify tools regardless of visible content: a substantive tool-only turn must
    # invalidate any older housekeeping fallback (so a two-turn-old housekeeping
    # narration isn't attributed to the preceding tool turn), and clear the mute flag a
    # prior housekeeping turn set, else _vprint suppresses this turn's tool progress.
    _all_housekeeping = all(
        tc.function.name in _HOUSEKEEPING_TOOLS for tc in assistant_message.tool_calls
    )
    if assistant_message.tool_calls and not _all_housekeeping:
        agent._last_content_with_tools = None
        agent._last_content_tools_all_housekeeping = False
        agent._mute_post_response = False

    # Content + tool_calls in one turn: keep the content as a fallback final response in
    # case the follow-up turn after tools is empty. Mute only when EVERY tool call is
    # post-response housekeeping; substantive tools keep output on.
    if turn_content and agent._has_content_after_think_block(turn_content):
        agent._last_content_with_tools = turn_content
        agent._last_content_tools_all_housekeeping = _all_housekeeping
        if _all_housekeeping and agent._has_stream_consumers():
            agent._mute_post_response = True
        elif agent._should_emit_quiet_tool_messages():
            clean = agent._strip_think_blocks(turn_content).strip()
            if clean:
                agent._vprint(f"  ┊ 💬 {clean}")

    # Pop thinking-only prefill message(s) before appending (same rationale as the
    # final-response path). Tool calls after a prefill recovery reset the prefill
    # counter, so each tool-call success is a fresh start, not a cumulative burn.
    _had_prefill = False
    while messages and isinstance(messages[-1], dict) and messages[-1].get("_thinking_prefill"):
        messages.pop()
        _had_prefill = True
    if _had_prefill:
        agent._thinking_prefill_retries = 0
        agent._empty_content_retries = 0
    # Re-arm the post-tool nudge so it can fire on a LATER tool round; a landed tool call
    # recovers any dropped-tool-call stall, so refresh that budget per stall.
    agent._post_tool_empty_retried = False
    agent._dropped_toolcall_retries = 0

    previous_msg = messages[-1] if messages else None
    current_interim_visible = agent._interim_assistant_visible_text(assistant_msg)
    duplicate_previous_interim = (
        bool(current_interim_visible)
        and isinstance(previous_msg, dict)
        and previous_msg.get("role") == "assistant"
        and previous_msg.get("finish_reason") == "incomplete"
        and agent._interim_assistant_visible_text(previous_msg) == current_interim_visible
    )
    return assistant_msg, duplicate_previous_interim


def transcript_tail_ended_by_tool(history: Any) -> bool:
    """True when ``history`` ends with a turn a turn-ending tool round closed (``tool_ended_turn``): the tool
    rows answer an assistant row with visible text whose calls are all ``ends_turn`` tools and none failed.
    Such a tail is a finished turn, not an interrupted one (gateway auto-continue must not fire)."""
    if not isinstance(history, list) or not history:
        return False
    from agent.display import _detect_tool_failure
    from tools.registry import registry
    i = len(history) - 1
    results = []
    while i >= 0 and isinstance(history[i], dict) and history[i].get("role") == "tool":
        results.append(history[i])
        i -= 1
    if not results or i < 0 or not isinstance(history[i], dict):
        return False
    row = history[i]
    content = row.get("content")
    if (row.get("role") != "assistant" or not row.get("tool_calls") or not isinstance(content, str)
            or not content.strip()):
        return False
    for call in row.get("tool_calls") or []:
        name = str(((call or {}).get("function") or {}).get("name") or "")
        entry = registry.get_entry(name) if name else None
        if entry is None or not getattr(entry, "ends_turn", False):
            return False
    return not any(_detect_tool_failure(str(r.get("name") or ""), r.get("content"))[0] for r in results)
