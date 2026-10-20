"""Per-run tool profile (iollo fork): a caller may run one turn with a smaller tool set.

``POST /v1/runs`` may carry ``tool_profile: {"name": "light", "tools": [...], "skills": false}`` (without
``tools``: every tool, the run is only named in its report). The agent is
built exactly as without it: the stored system prompt, the session's tool pin, compression rebuilds and every
persisted byte stay those of the full tool set. Only the request sent to the provider is projected
(``agent/turn_api_request.py``): ``tools[]`` keeps the named tools (in the session's order) plus any tool
registered ``lifts_tool_profile=True``, and with ``skills`` false the ``## Skills`` index block is cut out of the
system message. The text before it (identity, guidance, caller context) stays byte-identical.

Without a profile, ``lifts_tool_profile`` tools are left out of every request. With one, a call to such a tool, or
to any tool the agent has that the profile does not name, lifts the profile for the rest of the turn
(``lift_tool_profile_for_calls``): that call runs as usual and the next request carries every tool and the skills
index. A wrong light guess costs one model call, never a lost reply. The run reports the profile it asked for and
whether it was lifted (``tool_profile_report``).

A ``tools`` entry may also be a glob (``*`` only, e.g. ``mcp__notion__*``), resolved against the session's real
tools on every request, including its scoped pre-assembly catalog when Tool Search has deferred a requested
plugin/MCP tool. Only that request's schema and dispatch allowlist are materialized; the stored pin is unchanged.
A pattern that matches nothing is ignored. ``read_tools`` (globs or names) adds a
connector's READ tools only: an MCP tool whose discovery-time ``readOnlyHint`` is exactly True and whose own name
carries no write verb; anything else (no annotation, utilities, built-ins) stays with every tool. The read tools are
bounded per request (``MAX_READ_TOOLS`` tools, ``MAX_READ_CHARS`` schema characters, about 4k tokens): over either
bound none of them is sent (all or nothing, so the request stays stable) and the report says so. Calls run through
the same handlers as ever, so approvals and the MCP trust gate apply unchanged.

``hosted_tools`` (iollo, 2026-10-04): provider-executed tools added to this run's requests while the profile holds,
today only OpenAI's ``{"type": "web_search"}`` (optional ``search_context_size``, an approximate ``user_location``
and ``filters.allowed_domains``). They are appended after the projected function tools, last and in the caller's
order, replacing a client function of the same name for that request (they collide on the endpoint), only on
the native OpenAI Responses transport (never the ChatGPT Codex backend, Copilot or xAI, never chat
completions, so a Claude fallback or a worker never sees them). Nothing about them is persisted: ``web_search_call``
output items are not replayed (store=False), so a later turn without the tool, or on another model, replays plain
text. A lifted profile drops them with the rest of the projection. The run report counts the searches the provider
ran (``hosted_calls``), because the provider bills them per call on top of tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import json
import logging
import re
import sys
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

MAX_PROFILE_TOOLS = 200
MAX_NAME_LENGTH = 64
MAX_NOTE_LENGTH = 600
_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
# A glob: ``*`` wildcards only, after a literal prefix of at least 4 characters (never ``*`` or ``mc*`` alone).
_GLOB_RE = re.compile(r"^[A-Za-z0-9_.:-]{4,}[A-Za-z0-9_.:*-]*$")
MAX_READ_PATTERNS = 20
MAX_READ_TOOLS = 12
MAX_READ_CHARS = 16_000
MAX_HOSTED_TOOLS = 2
MAX_ALLOWED_DOMAINS = 20
_HOSTED_TOOL_TYPES = frozenset({"web_search"})
_SEARCH_CONTEXT_SIZES = frozenset({"low", "medium", "high"})
_LOCATION_FIELDS = {"country": 2, "city": 64, "region": 64, "timezone": 64}
_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
_DOMAIN_RE = re.compile(r"^[a-z0-9.-]{1,253}$")
# A read tool whose own name (not its server's) carries one of these words is never sent as a read tool, whatever its
# annotation says (defence in depth against a wrong readOnlyHint; the call-time gates apply either way).
_WRITE_WORDS = frozenset((
    "add", "append", "approve", "archive", "assign", "book", "cancel", "clear", "close", "comment", "complete",
    "create", "delete", "deploy", "disable", "duplicate", "edit", "enable", "execute", "import", "insert", "invite",
    "merge", "modify", "move", "pay", "post", "publish", "purchase", "put", "reject", "remove", "rename", "reply",
    "reset", "restart", "restore", "revoke", "run", "save", "schedule", "send", "set", "share", "start", "stop",
    "submit", "transfer", "trigger", "update", "upload", "upsert", "write"))

# The block ``agent.prompt_builder._render_skills_block`` renders (non-oneshot variant), plus its optional
# ``[names only]`` note. Matched from its heading to its fixed closing sentence.
_SKILLS_HEAD = "## Skills\nBefore replying, scan the skills below."
_SKILLS_TAIL = "Only proceed without loading a skill if genuinely none are relevant to the task."
_SKILLS_HIDDEN_NOTE_RE = re.compile(
    r"\n\(Categories marked \[names only\] are outside the current coding context, so their descriptions are "
    r"omitted — the skills work normally and load with skill_view\(name\) as usual\.\)")


@dataclass(frozen=True)
class ToolProfile:
    name: str
    tools: Optional[frozenset]  # None: every tool (a named, reported run without projection)
    skills: bool = True
    # Appended to each profile-lifting tool's description in this run's requests: what exists only with every
    # tool (the caller keeps it stable, so the profile's prefix stays cached).
    note: str = ""
    patterns: tuple = ()        # globs from ``tools``, resolved per request
    read_patterns: tuple = ()   # ``read_tools``: names/globs whose read-only MCP tools are sent, within the budget
    hosted_tools: tuple = ()    # provider-executed tools (validated dicts) appended on the native Responses route


def normalize_tool_profile(value: Any) -> Optional[ToolProfile]:
    """Request value -> profile; ``None``/absent -> ``None`` (every tool). Raises ``ValueError`` when invalid."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("'tool_profile' must be an object")
    name = value.get("name", "custom")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError("'tool_profile.name' must be a short identifier")
    tools = value.get("tools")
    if tools is None:
        if value.get("skills", True) is not True or value.get("read_tools") or value.get("hosted_tools"):
            raise ValueError("'tool_profile' without tools keeps every tool and the skills index")
        return ToolProfile(name=name, tools=None)
    if not isinstance(tools, list) or not tools or len(tools) > MAX_PROFILE_TOOLS:
        raise ValueError(f"'tool_profile.tools' must be a list of 1-{MAX_PROFILE_TOOLS} tool names")
    if any(not _valid_entry(t) for t in tools):
        raise ValueError("'tool_profile.tools' must contain tool names or globs")
    reads = value.get("read_tools", [])
    if not isinstance(reads, list) or len(reads) > MAX_READ_PATTERNS or any(not _valid_entry(t) for t in reads):
        raise ValueError(f"'tool_profile.read_tools' must be a list of at most {MAX_READ_PATTERNS} names or globs")
    skills = value.get("skills", True)
    if not isinstance(skills, bool):
        raise ValueError("'tool_profile.skills' must be a boolean")
    note = value.get("note", "")
    if not isinstance(note, str) or len(note) > MAX_NOTE_LENGTH:
        raise ValueError(f"'tool_profile.note' must be a string of at most {MAX_NOTE_LENGTH} characters")
    return ToolProfile(name=name, tools=frozenset(t for t in tools if "*" not in t), skills=skills,
                       note=" ".join(note.split()), patterns=tuple(dict.fromkeys(t for t in tools if "*" in t)),
                       read_patterns=tuple(dict.fromkeys(reads)),
                       hosted_tools=_normalize_hosted_tools(value.get("hosted_tools", [])))


def _normalize_hosted_tools(value: Any) -> tuple:
    """``hosted_tools`` -> a tuple of clean dicts (only the known fields, rebuilt). Raises ``ValueError``."""
    if not isinstance(value, list) or len(value) > MAX_HOSTED_TOOLS:
        raise ValueError(f"'tool_profile.hosted_tools' must be a list of at most {MAX_HOSTED_TOOLS} tools")
    out = []
    for tool in value:
        if not isinstance(tool, dict) or tool.get("type") not in _HOSTED_TOOL_TYPES:
            raise ValueError("'tool_profile.hosted_tools' entries must be {\"type\": \"web_search\", ...}")
        unknown = set(tool) - {"type", "search_context_size", "user_location", "filters"}
        if unknown:
            raise ValueError(f"'tool_profile.hosted_tools' has unknown fields: {sorted(unknown)}")
        clean: dict = {"type": tool["type"]}
        size = tool.get("search_context_size")
        if size is not None:
            if size not in _SEARCH_CONTEXT_SIZES:
                raise ValueError("'search_context_size' must be low, medium or high")
            clean["search_context_size"] = size
        location = tool.get("user_location")
        if location is not None:
            clean["user_location"] = _normalize_location(location)
        filters = tool.get("filters")
        if filters is not None:
            domains = filters.get("allowed_domains") if isinstance(filters, dict) else None
            if (not isinstance(filters, dict) or set(filters) != {"allowed_domains"} or not isinstance(domains, list)
                    or not 0 < len(domains) <= MAX_ALLOWED_DOMAINS
                    or any(not isinstance(d, str) or not _DOMAIN_RE.match(d) for d in domains)):
                raise ValueError("'filters' must be {\"allowed_domains\": [lower-case domains]}")
            clean["filters"] = {"allowed_domains": list(dict.fromkeys(domains))}
        out.append(clean)
    return tuple(out)


def _normalize_location(value: Any) -> dict:
    """An approximate location only: country (ISO 3166-1 alpha-2), city, region, IANA timezone; nothing finer."""
    if not isinstance(value, dict) or value.get("type", "approximate") != "approximate":
        raise ValueError("'user_location' must be an approximate location")
    unknown = set(value) - {"type", *_LOCATION_FIELDS}
    if unknown:
        raise ValueError(f"'user_location' has unknown fields: {sorted(unknown)}")
    clean = {"type": "approximate"}
    for key, limit in _LOCATION_FIELDS.items():
        field = value.get(key)
        if field is None:
            continue
        if not isinstance(field, str) or not field.strip() or len(field) > limit:
            raise ValueError(f"'user_location.{key}' must be a short string")
        if key == "country" and not _COUNTRY_RE.match(field):
            raise ValueError("'user_location.country' must be an ISO 3166-1 alpha-2 code")
        clean[key] = field.strip()
    return clean


def hosted_tools_for_request(agent: Any) -> List[dict]:
    """Fresh copies of the active profile's hosted tools for this request, or ``[]``: only while the profile holds
    and only on the native OpenAI Responses transport (not the ChatGPT Codex backend, Copilot or xAI)."""
    profile = active_tool_profile(agent)
    if profile is None or not profile.hosted_tools or getattr(agent, "api_mode", None) != "codex_responses":
        return []
    try:
        if agent._is_codex_backend() or agent._is_copilot_url():
            return []
    except Exception:
        return []
    if "x.ai" in str(getattr(agent, "base_url", "") or "").lower():
        return []
    return [json.loads(json.dumps(t)) for t in profile.hosted_tools]


def without_shadowed_functions(tools: List[Any], hosted: List[dict]) -> List[Any]:
    """``tools`` without a function tool named like one of ``hosted`` (e.g. the client ``web_search``)."""
    names = {t.get("type") for t in hosted}
    return [t for t in tools if not (_tool_name(t) in names and (not isinstance(t, dict) or t.get("type") == "function"))]


def note_hosted_calls(agent: Any, response: Any) -> None:
    """Count the provider-run searches in one Responses reply (``web_search_call`` output items) for the run report.
    Only for a run that asked for hosted tools; never raises."""
    requested = getattr(agent, "_tool_profile_requested", None)
    if not isinstance(requested, ToolProfile) or not requested.hosted_tools:
        return
    try:
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        n = 0
        for item in output or []:
            kind = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            n += kind == "web_search_call"
        if n:
            counts = getattr(agent, "_tool_profile_hosted_calls", None) or {}
            counts["web_search"] = counts.get("web_search", 0) + n
            agent._tool_profile_hosted_calls = counts
    except Exception:
        logger.debug("hosted call count failed", exc_info=True)


def _valid_entry(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > MAX_NAME_LENGTH:
        return False
    return bool(_GLOB_RE.match(value)) if "*" in value else bool(_NAME_RE.match(value))


def set_tool_profile(agent: Any, profile: Optional[ToolProfile]) -> None:
    """Declare this run's profile on a freshly built agent (before ``run_conversation``). A profile without
    tools projects nothing; it only names the run in its report."""
    _clear_profile_dispatch_names(agent)
    agent._tool_profile = profile if profile is not None and profile.tools is not None else None
    agent._tool_profile_requested = profile
    agent._tool_profile_lifted = None
    agent._tool_profile_allowed = None
    agent._tool_profile_reads = None
    agent._tool_profile_hosted_calls = None


def active_tool_profile(agent: Any) -> Optional[ToolProfile]:
    profile = getattr(agent, "_tool_profile", None)
    return profile if isinstance(profile, ToolProfile) else None


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
        if isinstance(tool.get("name"), str):
            return tool["name"]
    return ""


def _lifting_names(names: Iterable[str]) -> set:
    from tools.registry import registry
    out = set()
    for name in names:
        entry = registry.get_entry(name) if name else None
        if entry is not None and getattr(entry, "lifts_tool_profile", False):
            out.add(name)
    return out


def project_tools(agent: Any, tools: Optional[List[Any]]) -> Optional[List[Any]]:
    """The tools this request may declare; returns the same list object when nothing is removed."""
    profile = active_tool_profile(agent)
    if profile is None:
        _clear_profile_dispatch_names(agent)
        if not tools:
            return tools
        names = [_tool_name(t) for t in tools]
        lifting = _lifting_names(names)
        keep = [t for t, n in zip(tools, names) if n not in lifting]
        return tools if len(keep) == len(tools) else keep
    tools = list(tools or [])
    # Tool Search replaces deferred plugin/MCP schemas with its three bridge
    # tools during agent setup. A per-run profile naming one of those real
    # tools must project the original, session-scoped schema directly; the
    # bridge is for discovery, not a prerequisite for an explicit choice.
    catalog = _profile_catalog_tools(agent, profile, tools)
    catalog_names = [_tool_name(t) for t in catalog]
    catalog_lifting = _lifting_names(catalog_names)
    allowed, reads = _resolve(profile, catalog, catalog_names, catalog_lifting)
    agent._tool_profile_allowed = allowed
    if reads is not None:
        agent._tool_profile_reads = reads
    keep = [_with_note(t, profile.note) if n in catalog_lifting else t
            for t, n in zip(catalog, catalog_names) if n in allowed or n in catalog_lifting]
    # Validation and direct dispatch use this request-local allowlist. Do not
    # append to agent.tools: that is the session's stored/full pin and must
    # remain unchanged across profile requests.
    base = {_tool_name(t) for t in getattr(agent, "tools", []) or []}
    _set_profile_dispatch_names(agent, {n for n in allowed | catalog_lifting if n not in base})
    return keep


def _clear_profile_dispatch_names(agent: Any) -> None:
    _set_profile_dispatch_names(agent, set())


def _set_profile_dispatch_names(agent: Any, names: set[str]) -> None:
    valid = getattr(agent, "valid_tool_names", None)
    previous = getattr(agent, "_tool_profile_dispatch_names", set())
    if valid is not None:
        valid.difference_update(previous)
        added = names - valid
        valid.update(added)
    else:
        added = set()
    agent._tool_profile_dispatch_names = set(added)


def _profile_catalog_tools(agent: Any, profile: ToolProfile, visible_tools: List[Any]) -> List[Any]:
    """Visible schemas plus profile-selected/read and profile-lifting schemas from the scoped catalog.

    ``get_tool_definitions(..., skip_tool_search_assembly=True)`` is the same
    allowlisted pre-assembly catalog used by Tool Search itself. We materialize
    only names requested by this profile or its read patterns, plus the normal
    lifting tools; retain the same one-shot and side-agent exclusions as agent
    initialization.
    """
    by_name = {_tool_name(tool): tool for tool in visible_tools if _tool_name(tool)}
    wanted = set(profile.tools or ())
    patterns = (*profile.patterns, *profile.read_patterns)
    for tool in visible_tools:
        name = _tool_name(tool)
        if name and (name in wanted or _matches(name, patterns)):
            wanted.add(name)
    try:
        from tools.tool_search import BRIDGE_TOOL_NAMES
        search_is_active = bool(set(by_name) & BRIDGE_TOOL_NAMES)
    except Exception:
        search_is_active = False
    raw_needed = bool(search_is_active or patterns or wanted - set(by_name))
    if not raw_needed:
        return list(by_name.values())
    try:
        import model_tools
        raw = model_tools.get_tool_definitions(
            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            quiet_mode=True,
            skip_tool_search_assembly=True,
        ) or []
        from agent.oneshot_footprint import prune_oneshot_tools
        raw = prune_oneshot_tools(raw)
        from tools.connectors.turn import side_agent_tool_drops
        drops = side_agent_tool_drops(agent)
        from tools.registry import registry
        for tool in raw:
            name = _tool_name(tool)
            if not name or name in drops:
                continue
            entry = registry.get_entry(name)
            if (name in wanted or _matches(name, patterns)
                    or (entry is not None and getattr(entry, "lifts_tool_profile", False))):
                by_name.setdefault(name, tool)
    except Exception as exc:
        # Fail this request visibly rather than silently producing an empty
        # projection that looks like a successful no-tools profile.
        logger.exception("deferred tool profile catalog unavailable")
        raise RuntimeError("Unable to resolve this run's scoped tool profile") from exc
    return list(by_name.values())


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(name, p) if "*" in p else name == p for p in patterns)


def _resolve(profile: ToolProfile, tools: List[Any], names: List[str], lifting: set):
    """The names this request may declare under ``profile`` (a frozenset) and the read-tools report (None without
    ``read_tools``): named tools, glob matches, then read tools within the budget."""
    allowed = {n for n in names if n and n not in lifting and (n in profile.tools or _matches(n, profile.patterns))}
    if not profile.read_patterns:
        return frozenset(allowed), None
    counts = {p: 0 for p in profile.read_patterns}
    reads, chars = [], 0
    for tool, n in zip(tools, names):
        if not n or n in allowed or n in lifting:
            continue
        hit = next((p for p in profile.read_patterns if _matches(n, (p,))), None)
        if hit is None or not _read_only_tool(n):
            continue
        counts[hit] += 1
        reads.append(n)
        chars += len(json.dumps(tool, separators=(",", ":"), ensure_ascii=False, default=str))
    status = "ok" if reads else "none"
    if len(reads) > MAX_READ_TOOLS or chars > MAX_READ_CHARS:
        status = "over_budget"
    else:
        allowed.update(reads)
    return frozenset(allowed), {"status": status, "tools": len(reads), "chars": chars, "patterns": counts}


def _name_words(raw: str) -> set:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(raw))
    return {w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w}


def _read_only_tool(name: str) -> bool:
    """True only for a registered MCP tool whose server marked it ``readOnlyHint: true`` at discovery (the record the
    call-time trust gate reads) and whose own name has no write verb. Fails closed."""
    if "tools.mcp_tool" not in sys.modules:   # nothing imported the MCP client, so no MCP tool is registered
        return False
    try:
        from tools import mcp_tool as core
        from tools.mcp_tool_schema import mcp_prefixed_tool_name
        from tools.mcp_tool_scope import _resolve_server_key
        server = core._mcp_tool_server_names.get(name)
        if not server:
            return False
        hints = core._tool_read_only_hints.get(_resolve_server_key(server), {})
        for raw, read_only in list(hints.items()):
            if read_only is True and mcp_prefixed_tool_name(server, raw) == name:
                return not (_name_words(raw) & _WRITE_WORDS)
    except Exception:
        logger.debug("read tool check failed for %s", name, exc_info=True)
    return False


def _with_note(tool: Any, note: str) -> Any:
    """A copy of ``tool`` whose description ends with ``note`` (the original definition is never mutated)."""
    fn = tool.get("function") if isinstance(tool, dict) else None
    if not note or not isinstance(fn, dict):
        return tool
    description = str(fn.get("description") or "")
    return {**tool, "function": {**fn, "description": (description + " " + note).strip()}}


def strip_skills_block(text: str) -> str:
    """``text`` without the ``## Skills`` index block (and the blank line before it); unchanged without one."""
    if not isinstance(text, str):
        return text
    start = text.find(_SKILLS_HEAD)
    if start < 0:
        return text
    end = text.find(_SKILLS_TAIL, start)
    if end < 0:
        return text
    end += len(_SKILLS_TAIL)
    note = _SKILLS_HIDDEN_NOTE_RE.match(text, end)
    if note:
        end = note.end()
    before, after = text[:start], text[end:]
    if before.endswith("\n\n") and after.startswith("\n\n"):
        before = before[:-2]
    elif not before.strip():
        after = after.lstrip("\n")
    return before + after


def _strip_system_content(content: Any) -> Any:
    if isinstance(content, str):
        return strip_skills_block(content)
    if isinstance(content, list):
        out, changed = [], False
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text = strip_skills_block(part["text"])
                if text != part["text"]:
                    part, changed = {**part, "text": text}, True
            out.append(part)
        return out if changed else content
    return content


def project_system_message(agent: Any, api_messages: List[Any]) -> List[Any]:
    """``api_messages`` with the skills index cut from the leading system message when the profile drops skills.
    Returns a new list (the original message dicts are never mutated) or the same list when nothing changes."""
    profile = active_tool_profile(agent)
    if profile is None or profile.skills or not api_messages:
        return api_messages
    first = api_messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return api_messages
    content = _strip_system_content(first.get("content"))
    if content is first.get("content"):
        return api_messages
    return [{**first, "content": content}, *api_messages[1:]]


def lift_tool_profile_for_calls(agent: Any, tool_calls: Iterable[Any]) -> bool:
    """Lift the profile when a call names a profile-lifting tool or a tool the profile left out; True if lifted.
    Unknown names are left to the normal invalid-tool path."""
    profile = active_tool_profile(agent)
    if profile is None:
        return False
    valid = getattr(agent, "valid_tool_names", None) or set()
    names = []
    for tc in tool_calls or []:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) if fn is not None else None
        if isinstance(name, str):
            names.append(name)
    lifting = _lifting_names(names)
    allowed = getattr(agent, "_tool_profile_allowed", None)
    if not isinstance(allowed, frozenset):   # no request projected yet: names and globs only
        allowed = frozenset(n for n in names if n in profile.tools or _matches(n, profile.patterns))
    outside = [n for n in names if n in valid and (n not in allowed or n in lifting)]
    if not outside:
        return False
    reason = "escape" if any(n in lifting for n in outside) else "tool:" + outside[0]
    agent._tool_profile = None
    agent._tool_profile_lifted = reason
    logger.info("tool profile %s lifted for session %s (%s)", profile.name, getattr(agent, "session_id", ""), reason)
    return True


def normalize_ends_turn(value: Any) -> bool:
    """Request ``ends_turn`` -> bool (absent: False). Turn-ending tools end a turn only on runs that ask for it."""
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError("'ends_turn' must be a boolean")
    return value


def set_ends_turn(agent: Any, enabled: bool) -> None:
    agent._ends_turn_enabled = bool(enabled)


def normalize_skip_background_review(value: Any) -> bool:
    """Request ``skip_background_review`` -> bool (absent: False). A machine-initiated run in its own session (the
    home feed curation) asks for no post-turn memory/skill review, for this run only (``agent.skip_background_review``,
    the flag cron runs already set)."""
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError("'skip_background_review' must be a boolean")
    return value


def normalize_max_iterations(value: Any) -> Optional[int]:
    """Request ``max_iterations`` -> a per-run round cap (absent: None, the configured budget). It only lowers the
    configured ``max_turns``: a chat surface asks for a short loop, its budget hit forces the usual final answer."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("'max_iterations' must be a positive integer")
    return value


def tool_profile_report(agent: Any, api_calls: Any = None) -> Optional[dict]:
    """``{"name", "lifted", "api_calls"}`` for a run that asked for a profile, else ``None`` (wire shape
    unchanged). ``api_calls`` lets a caller compare input tokens per model call across profiles."""
    requested = getattr(agent, "_tool_profile_requested", None)
    if not isinstance(requested, ToolProfile):
        return None
    lifted = getattr(agent, "_tool_profile_lifted", None)
    calls = api_calls if isinstance(api_calls, int) and not isinstance(api_calls, bool) else 0
    report = {"name": requested.name, "lifted": lifted or "", "api_calls": calls}
    if requested.read_patterns:
        # {"status": ok | none | over_budget, "tools", "chars", "patterns": {pattern: read tools found}}
        reads = getattr(agent, "_tool_profile_reads", None)
        report["read_tools"] = reads if isinstance(reads, dict) else {
            "status": "none", "tools": 0, "chars": 0, "patterns": {p: 0 for p in requested.read_patterns}}
    if requested.hosted_tools:
        # {"web_search": provider-run searches across this run's requests} (billed per call by the provider)
        hosted = getattr(agent, "_tool_profile_hosted_calls", None)
        report["hosted_calls"] = dict(hosted) if isinstance(hosted, dict) else {}
    return report
