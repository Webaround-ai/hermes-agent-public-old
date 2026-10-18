"""``escalate``: ask a stronger model for advice, over the delegate_task child machinery.

Off unless ``delegation.escalate`` is configured (the tool is not even advertised otherwise)::

    delegation:
      tiers:
        opus_high: {model: anthropic/claude-opus-5.5, reasoning_effort: high}
        fable: {model: anthropic/claude-fable-5.1, reasoning_effort: high}
      escalate:
        tiers: [opus_high, fable]  # required: level 0 = first call in a turn, level 1 = any later call
                                   # (the last entry repeats); legacy `tier: opus_high` = a one-entry list
        max_per_task: 2        # optional, per main-agent turn; unset = unlimited
        max_per_day: 10        # optional, per profile per UTC day; unset = unlimited
        max_iterations: 30     # advisor's own iteration budget (default 30)

The advisor starts cold with a read-only toolset (web research, vision, past-session search — never
terminal, files, sends or anything needing an approval) and returns advice only; the main agent applies it.
Caps are enforced here, in code, before the advisor is launched.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Advisor toolsets: read-only by construction. Intersected with the parent's own toolsets.
ESCALATE_READONLY_TOOLSETS = ("web", "search", "vision", "session_search")
DEFAULT_MAX_ITERATIONS = 30

_lock = threading.Lock()
_task_counts: "OrderedDict[Tuple[str, str, str], int]" = OrderedDict()
_TASK_COUNT_LIMIT = 2048


def _int(raw: Any, default: int, floor: int = 0) -> int:
    try:
        return max(floor, int(raw))
    except (TypeError, ValueError):
        return default


def _cap(raw: Any) -> Optional[int]:
    """An optional cap: unset/null/malformed = ``None`` (unlimited); a number >= 0 is enforced."""
    if raw is None or raw == "" or isinstance(raw, bool):
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed delegation.escalate cap %r (treated as unlimited)", raw)
        return None


def escalate_config(cfg: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """Normalized ``delegation.escalate`` or ``None`` (feature off: missing, malformed, or no known tier).
    ``tiers`` is the ordered level list; unknown tier names are dropped."""
    if cfg is None:
        from tools.delegate_tool_config import _load_config
        cfg = _load_config()
    raw = cfg.get("escalate") if isinstance(cfg, dict) else None
    if not isinstance(raw, dict):
        return None
    from tools.delegate_tool_tiers import configured_tiers
    raw_tiers = raw.get("tiers")
    if raw_tiers is None:
        raw_tiers = [raw.get("tier")]
    elif not isinstance(raw_tiers, (list, tuple)):
        raw_tiers = [raw_tiers]
    known = configured_tiers(cfg)
    names = [str(t or "").strip() for t in raw_tiers]
    tiers = [n for n in names if n in known]
    if len(tiers) != len([n for n in names if n]):
        logger.warning("delegation.escalate names tiers that are not configured: %s",
                       [n for n in names if n and n not in known])
    if not tiers:
        return None
    return {
        "tiers": tiers,
        "max_per_task": _cap(raw.get("max_per_task")),
        "max_per_day": _cap(raw.get("max_per_day")),
        "max_iterations": _int(raw.get("max_iterations"), DEFAULT_MAX_ITERATIONS, floor=1),
    }


def check_escalate_requirements() -> bool:
    try:
        return escalate_config() is not None
    except Exception:
        return False


# ── caps ────────────────────────────────────────────────────────────────────

def _day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _usage_path():
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state" / "escalate_usage.json"


def _read_day_count() -> int:
    try:
        data = json.loads(_usage_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return _int(data.get("count"), 0) if isinstance(data, dict) and data.get("date") == _day() else 0


def _write_day_count(count: int) -> None:
    path = _usage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"date": _day(), "count": count}), encoding="utf-8")
    tmp.replace(path)


def _task_key(parent_agent) -> Tuple[str, str, str]:
    from hermes_constants import hermes_home_key
    session = str(getattr(parent_agent, "session_id", "") or "")
    turn = str(getattr(parent_agent, "_current_turn_id", "") or "")
    return (str(hermes_home_key()), session, turn or session)


def usage(parent_agent) -> Tuple[int, int]:
    """``(used_this_task, used_today)``."""
    with _lock:
        return _task_counts.get(_task_key(parent_agent), 0), _read_day_count()


def _reserve(parent_agent, esc: Dict[str, Any]) -> Tuple[Optional[str], int]:
    """Count one escalation and return ``(None, level)`` (level = escalations already made this turn), or
    ``(refusal text, level)`` when a configured cap is reached (nothing counted). Unset caps never refuse."""
    key = _task_key(parent_agent)
    with _lock:
        used_task, used_day = _task_counts.get(key, 0), _read_day_count()
        if esc["max_per_task"] is not None and used_task >= esc["max_per_task"]:
            return (f"Escalation limit for this task reached ({used_task}/{esc['max_per_task']}). "
                    "Continue with your own best judgment, or tell the user what you are unsure about."), used_task
        if esc["max_per_day"] is not None and used_day >= esc["max_per_day"]:
            return (f"Daily escalation limit reached ({used_day}/{esc['max_per_day']}). "
                    "Continue with your own best judgment, or tell the user what you are unsure about."), used_task
        _task_counts[key] = used_task + 1
        _task_counts.move_to_end(key)
        while len(_task_counts) > _TASK_COUNT_LIMIT:
            _task_counts.popitem(last=False)
        try:
            _write_day_count(used_day + 1)
        except OSError:
            logger.warning("escalate: could not persist the daily counter", exc_info=True)
    return None, used_task


def level_tier(esc: Dict[str, Any], level: int) -> str:
    """Tier for escalation *level* (0 = first call in the turn); the last entry repeats."""
    tiers = esc["tiers"]
    return tiers[min(max(0, level), len(tiers) - 1)]


def _refund(parent_agent) -> None:
    key = _task_key(parent_agent)
    with _lock:
        if _task_counts.get(key, 0) > 0:
            _task_counts[key] -= 1
        with_day = _read_day_count()
        if with_day > 0:
            try:
                _write_day_count(with_day - 1)
            except OSError:
                pass


def _reset_counters_for_tests() -> None:
    with _lock:
        _task_counts.clear()


# ── tool ────────────────────────────────────────────────────────────────────

_ADVISOR_BRIEF = (
    "You are an advisor to another AI agent that is stuck or facing a hard decision. You are READ-ONLY: you "
    "cannot change files, run commands, send messages or act for anyone; the other agent applies your advice. "
    "Research only if it genuinely helps. Answer with: the decision you recommend, why (briefly), and the "
    "concrete next steps the agent should take. If the brief is missing something essential, say what and "
    "give your best recommendation anyway."
)


def _compose(question: str, context: str, constraints: str, wanted: str) -> Tuple[str, str]:
    goal = f"Advise: {wanted or question}"
    parts = [_ADVISOR_BRIEF, f"Question:\n{question}"]
    if context:
        parts.append(f"Context (what was tried, exact errors):\n{context}")
    if constraints:
        parts.append(f"Constraints:\n{constraints}")
    if wanted:
        parts.append(f"Decision needed:\n{wanted}")
    return goal, "\n\n".join(parts)


def escalate(question: str = "", context: str = "", constraints: str = "", wanted: str = "",
             parent_agent=None) -> str:
    """Run one advice-only child on the configured escalation tier and return its answer as JSON."""
    if parent_agent is None:
        return tool_error("escalate requires a parent agent context.")
    question, context = str(question or "").strip(), str(context or "").strip()
    constraints, wanted = str(constraints or "").strip(), str(wanted or "").strip()
    if not question:
        return tool_error("escalate needs a 'question'. The advisor starts cold: give the question, what you "
                          "tried with exact errors in 'context', your 'constraints', and the decision 'wanted'.")
    if getattr(parent_agent, "_delegate_depth", 0) > 0:
        return tool_error("escalate is only available to the main agent.")
    from tools.delegate_tool_config import _load_config, _resolve_delegation_credentials
    cfg = _load_config()
    esc = escalate_config(cfg)
    if esc is None:
        return tool_error("Escalation is not configured (delegation.escalate).")

    from tools.delegate_tool_tiers import build_task_routes, route_summary
    try:
        base_creds = _resolve_delegation_credentials(cfg, parent_agent)
    except ValueError as exc:
        return tool_error(str(exc))

    # Caps are checked before launching the advisor.
    refusal, level = _reserve(parent_agent, esc)
    if refusal:
        return json.dumps({"escalated": False, "reason": refusal}, ensure_ascii=False)
    tier = level_tier(esc, level)
    try:
        route = build_task_routes([{"goal": question, "tier": tier}], cfg, base_creds)[0]
    except ValueError as exc:
        _refund(parent_agent)
        return tool_error(str(exc))
    goal, brief = _compose(question, context, constraints, wanted)
    try:
        result = _run_advisor(parent_agent, goal, brief, route, base_creds, esc)
    except ValueError as exc:
        _refund(parent_agent)
        return tool_error(str(exc))
    entry = (result.get("results") or [{}])[0]
    out: Dict[str, Any] = {
        "escalated": True, "status": entry.get("status"), "advice": entry.get("summary") or "",
        "advised_by": route["model"], "level": level,
        **{k: v for k, v in (route_summary(route, 0) or {}).items() if k in ("tier", "model", "reasoning_effort")},
    }
    if entry.get("error"):
        out["error"] = entry["error"]
    for key in ("cost_usd", "duration_seconds"):
        if key in entry:
            out[key] = entry[key]
    out["note"] = f"Advice from {route['model']} (level {level}), advice only: nothing was changed. Apply it yourself if you agree."
    return json.dumps(out, ensure_ascii=False)


def _run_advisor(parent_agent, goal: str, brief: str, route: Dict[str, Any], base_creds: Dict[str, Any],
                 esc: Dict[str, Any]) -> Dict[str, Any]:
    from tools.delegate_tool_dispatch import _Batch, _capture_origin, _execute_and_aggregate
    from tools.delegate_tool_results import _build_child_preserving_parent_tools
    from tools.delegate_tool_tiers import route_summary
    from tools.delegation_live_log import create_live_transcripts
    creds = route["creds"]
    task = {"goal": goal, "context": brief}
    overall_start = time.monotonic()
    facts = [route_summary(route, 0)]
    live_deleg_id, live_writers, live_paths = create_live_transcripts(
        [task], None, model=route["model"], provider=creds.get("provider"), task_routes=facts)
    origin = _capture_origin()
    child = _build_child_preserving_parent_tools(
        task_index=0, goal=goal, context=brief, toolsets=list(ESCALATE_READONLY_TOOLSETS),
        readonly_toolsets=list(ESCALATE_READONLY_TOOLSETS), model=route["model"],
        max_iterations=esc["max_iterations"], task_count=1, parent_agent=parent_agent,
        override_provider=creds["provider"], override_base_url=creds["base_url"],
        override_api_key=creds["api_key"], override_api_mode=creds["api_mode"],
        override_request_overrides=creds.get("request_overrides"),
        override_acp_command=creds.get("command"), override_acp_args=creds.get("args"),
        reasoning_effort=route.get("reasoning_effort"), tier=route.get("tier"),
        routing_cfg={"fallback_providers": []},  # advice must come from the selected tier
    )
    writer = live_writers[0] if live_writers else None
    if writer is not None:
        from tools.delegation_live_log import wrap_progress_callback
        child.tool_progress_callback = wrap_progress_callback(getattr(child, "tool_progress_callback", None), writer)
        child._live_transcript_path = str(writer.path)
    if live_deleg_id:
        setattr(child, "_delegation_id", live_deleg_id)
    batch = _Batch([task], [(0, task, child)], parent_agent, base_creds, None, "leaf", 1,
                   live_deleg_id, live_writers, live_paths, *origin, overall_start, task_routes=facts)
    return _execute_and_aggregate(batch)


ESCALATE_SCHEMA = {
    "name": "escalate",
    "description": (
        "Ask a stronger model for ADVICE on a hard decision or a problem you could not solve after real "
        "attempts. Give the concrete difficulty and failed attempts. Paid advice may be capped, so use it rarely. "
        "Your own model and reasoning stay unchanged. The advisor starts "
        "COLD — it sees nothing of this conversation — so write a complete brief: the question, what you "
        "tried with exact errors, your constraints, and the decision you need. It cannot act: it returns "
        "advice, and you apply it. If a configured cap is reached, continue with your own best judgment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question or problem, self-contained."},
            "context": {"type": "string", "description": "What you tried and what happened: exact error text, "
                                                         "file paths, relevant facts. The advisor knows nothing else."},
            "constraints": {"type": "string", "description": "Limits the answer must respect (budget, tools, "
                                                             "user preferences, deadlines)."},
            "wanted": {"type": "string", "description": "The specific decision or answer you need back."},
        },
        "required": ["question", "context", "wanted"],
    },
}

registry.register(
    name="escalate",
    toolset="escalate",
    schema=ESCALATE_SCHEMA,
    handler=lambda args, **kw: escalate(
        question=args.get("question", ""), context=args.get("context", ""),
        constraints=args.get("constraints", ""), wanted=args.get("wanted", ""),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_escalate_requirements,
    emoji="🧭",
)
