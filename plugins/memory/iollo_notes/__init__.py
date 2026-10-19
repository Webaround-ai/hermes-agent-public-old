"""Connected Mac recall (the only thing this memory provider does).

Activate with ``memory.provider: iollo_notes``. Hermes' native memory (MEMORY.md / USER.md through the built-in
``memory`` tool) is the only memory; this provider offers one tool, ``recall``, which searches the connected
Mac's own index through ``/relay/device-tool``, sending the query rather than notes. It injects nothing into the
prompt, prefetches nothing and mirrors no writes. When the Mac is disconnected, recall is off or the Mac does not
answer, recall says so plainly.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error, tool_result

logger = logging.getLogger(__name__)

PROVIDER_NAME = "iollo_notes"
RECALL_TIMEOUT_S = 10.0
RECALL_MAC_DEADLINE_S = 20               # how long recall waits for the Mac's own index
RECALL_DEFAULT_LIMIT, RECALL_MAX_LIMIT = 10, 20
DEFAULT_RELAY_URL = "http://instinct-control.internal:8080"
MAC_RECALL_TOOL = "mac_recall_local"
# What recall answers when it cannot search the Mac: the reason, said plainly, never an empty result.
MAC_UNAVAILABLE = {
    "not_connected": "The owner's Mac is not connected, so recall cannot search it now. Say that, rather than that "
                     "it does not exist; it works again once the Mac is awake with iollo running.",
    "recall_off": "Recall is off on the owner's Mac, so it cannot be searched. Say that; the owner can turn it on "
                  "in the iollo menu bar app.",
    "unreachable": "The owner's Mac did not answer, so recall could not search it. Say that, rather than that it "
                   "does not exist.",
}


RECALL_SCHEMA = {
    "name": "recall",
    "description": (
        "Find something on the owner's Mac by what it is about: a document, a PDF page, a screenshot, an email, a "
        "message, a note, an event, a visited page. Searches the connected Mac's own index; when the Mac is "
        "not connected or does not answer, it says so. Answers where each hit is (folder and page, who sent it "
        "and when); fetch the thing itself with mac_call (file_send, mail_read, notes_read). Use it before saying "
        "you cannot find something of the owner's."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What it is about, in plain words (any language)."},
            "sources": {"type": "array", "items": {"type": "string"},
                        "description": "Only these sources: files, mail, messages, notes, calendar, reminders, "
                                       "contacts, history, ai_tools."},
            "since": {"type": "string", "description": "Only things from this time on (RFC 3339 or YYYY-MM-DD)."},
            "limit": {"type": "integer", "description": "Maximum hits (default 10, at most 20)."},
        },
        "required": ["query"],
    },
}


def _post_json(url: str, body: Any, headers: Dict[str, str], timeout: float) -> Any:
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - policy-set / relay URL
        return json.loads(response.read(4 * 1024 * 1024).decode("utf-8"))


def _get_json(url: str, headers: Dict[str, str], timeout: float) -> Any:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - relay URL
        return json.loads(response.read(1024 * 1024).decode("utf-8"))


def _secret(name: str) -> str:
    """A relay setting the gateway runs with (profile-scoped when the gateway multiplexes profiles)."""
    try:
        from gateway.platforms._shared import get_scoped_secret

        value = get_scoped_secret(name, "")
    except Exception:
        value = os.environ.get(name, "")
    return str(value or "").strip()


def _clock(seconds: Any) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return ""
    hours, rest = divmod(total, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def where_it_is(hit: Dict[str, Any]) -> str:
    """Where a recall hit is, in the owner's words: "Documents/Taxes/2025, lease.pdf, page 3", "message from Ana on
    2026-03-03", "email from Ana on 2026-03-03 ("Lease")", "note "Porto lease" in Notes", "calendar event "Dentist"
    on 2026-03-03", "visited example.com on 2026-03-03"."""
    source, kind = str(hit.get("source") or ""), str(hit.get("kind") or "")
    ref, title = str(hit.get("ref") or ""), str(hit.get("title") or "").strip()
    when = str(hit.get("when") or "")[:10]
    people = [str(p) for p in hit.get("people") or [] if str(p).strip()]
    who = f" from {people[0]}" if people else ""
    on = f" on {when}" if when else ""
    loc = hit.get("loc")
    if source == "files":
        folder, _, name = ref.rpartition("/")
        place = f"{folder}, {name}" if folder else (name or title or "a file")
        if loc not in (None, ""):
            if kind in ("pdf", "page", "document"):
                place += f", page {loc}"
            elif kind in ("audio", "video", "recording"):
                place += f", at {_clock(loc)}"
            elif kind in ("text", "code"):
                place += f", line {loc}"
        return place
    if source == "mail":
        return f"email{who}{on}" + (f' ("{title}")' if title else "")
    if source == "messages":
        return f"message{who}{on}"
    if source == "notes":
        return f'note "{title}" in Notes' if title else "a note in Notes"
    if source in ("calendar", "reminders"):
        what = "reminder" if source == "reminders" else "calendar event"
        return f'{what} "{title}"{on}' if title else f"{what}{on}"
    if source == "contacts":
        return f'contact "{title}"' if title else "a contact"
    if source == "history":
        host = urllib.parse.urlsplit(ref).hostname or ref
        return f"visited {host}{on}"
    if source == "ai_tools":
        return f"your own words in an AI tool{on}" + (f' ("{title}")' if title else "")
    return f"{source or 'Mac'}{on}" + (f' ("{title}")' if title else "")


def _one_line(text: str) -> str:
    """Collapse whitespace."""
    return " ".join((text or "").split())


class IolloNotesProvider(MemoryProvider):
    """Mac recall only; Hermes' native memory holds everything else."""

    def __init__(self, config: Optional[dict] = None):
        self._config = dict(config or {})

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        return None

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "recall":
            return tool_error(f"Unknown tool: {tool_name}")
        try:
            return self._tool_recall(args or {})
        except ValueError as exc:
            return tool_error(str(exc))
        except Exception as exc:
            logger.warning("iollo_notes recall failed: %s", type(exc).__name__)
            return tool_error("recall failed")

    # -- recall (the Mac's own index; the Mac is a tool) -----------------------------------------

    def _mac_search(self, query: str, sources: List[str], since: Optional[str], limit: int) -> Dict[str, Any]:
        """The connected Mac's own index through the control plane (the route mac_call uses). ``state``: searched,
        not_connected, recall_off, indexing or unreachable."""
        token = _secret("RELAY_TOKEN")
        if not token:
            return {"state": "not_connected", "hits": []}
        base = (_secret("RELAY_URL") or DEFAULT_RELAY_URL).rstrip("/")
        headers = {"X-Relay-Token": token}
        try:
            listing = _get_json(f"{base}/relay/device-tool", headers, RECALL_TIMEOUT_S)
        except (urllib.error.URLError, OSError, ValueError):
            return {"state": "unreachable", "hits": []}
        listing = listing if isinstance(listing, dict) else {}
        if not listing.get("connected"):
            return {"state": "not_connected", "hits": []}
        tools = {t.get("name") for t in listing.get("tools") or [] if isinstance(t, dict)}
        recall_cap = listing.get("recall") if isinstance(listing.get("recall"), dict) else {}
        if MAC_RECALL_TOOL not in tools or recall_cap.get("enabled") is not True:
            return {"state": "recall_off", "hits": []}
        args: Dict[str, Any] = {"query": query, "limit": limit}
        if sources:
            args["sources"] = sources
        if since:
            args["since"] = since
        device = listing.get("device") if isinstance(listing.get("device"), dict) else {}
        payload: Dict[str, Any] = {"tool": MAC_RECALL_TOOL, "args": args, "deadline_s": RECALL_MAC_DEADLINE_S}
        if device.get("id"):
            payload["device_id"] = device["id"]
        try:
            answer = _post_json(f"{base}/relay/device-tool", payload, headers, RECALL_MAC_DEADLINE_S + 10)
        except (urllib.error.URLError, OSError, ValueError):
            return {"state": "unreachable", "hits": [], "device_id": device.get("id")}
        answer = answer if isinstance(answer, dict) else {}
        data = answer.get("data") if isinstance(answer.get("data"), dict) else {}
        error = answer.get("error") if isinstance(answer.get("error"), dict) else {}
        state = ("searched" if answer.get("state") in ("done", "succeeded") else
                 error["code"] if error.get("code") in ("recall_off", "indexing") else "unreachable")
        hits = [h for h in data.get("hits") or [] if isinstance(h, dict)] if state in ("searched", "indexing") else []
        return {"state": state, "hits": hits, "device_id": device.get("id"), "indexed": data.get("indexed"),
                "total": data.get("total")}

    def _tool_recall(self, args: Dict[str, Any]) -> str:
        query = _one_line(str(args.get("query") or ""))[:500]
        if not query:
            raise ValueError("query is required")
        raw_sources = args.get("sources") if isinstance(args.get("sources"), list) else []
        sources = [s.strip() for s in raw_sources if isinstance(s, str) and s.strip()][:10]
        since = str(args.get("since") or "").strip() or None
        try:
            limit = max(1, min(int(args.get("limit") or RECALL_DEFAULT_LIMIT), RECALL_MAX_LIMIT))
        except (TypeError, ValueError):
            limit = RECALL_DEFAULT_LIMIT
        mac = self._mac_search(query, sources, since, limit)
        if mac["state"] not in ("searched", "indexing"):
            return tool_error(MAC_UNAVAILABLE.get(mac["state"], MAC_UNAVAILABLE["unreachable"]), mac=mac["state"])
        hits: List[Dict[str, Any]] = []
        seen = set()
        for hit in mac["hits"]:
            key = hit.get("ref")
            if key in seen:
                continue
            seen.add(key)
            hits.append({"from": "mac", "where": where_it_is(hit),
                         **{k: hit[k] for k in ("source", "kind", "ref", "title", "when", "people", "excerpt", "loc",
                                                "loc_end") if hit.get(k) not in (None, "", [])},
                         "device_id": mac.get("device_id")})
        hits = hits[:limit]
        indexed, total = mac.get("indexed"), mac.get("total")
        out: Dict[str, Any] = {"hits": hits, "count": len(hits), "mac": mac["state"]}
        if isinstance(indexed, int) and isinstance(total, int):
            out.update(indexed=indexed, total=total)
        if mac["state"] == "indexing" or (isinstance(indexed, int) and isinstance(total, int) and indexed < total):
            out["still_reading"] = True
            out["note"] = ("iollo is still reading the owner's Mac: if this is not what they meant, say "
                           "\"I'm still reading your Mac\" rather than that it does not exist.")
        elif not hits:
            out["note"] = "Nothing found on the owner's Mac for that."
            logger.info("recall miss: state=%s query_chars=%d sources=%d", mac["state"], len(query), len(sources))
        return tool_result(out)


def register(ctx) -> None:
    """Register the iollo_notes memory provider with the plugin system."""
    ctx.register_memory_provider(IolloNotesProvider())
