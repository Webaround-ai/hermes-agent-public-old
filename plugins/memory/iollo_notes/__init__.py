"""Typed local notes, profile injection and connected Mac recall.

Activate with ``memory.provider: iollo_notes``. The ``memory.iollo_notes.device`` setting
records provenance as ``box`` or ``mac:<id>``. Facts are filed and retrieved through the local
index in ``$HERMES_HOME``; consolidation and USER.md generation run on the box.

``recall`` searches the connected Mac's own index through ``/relay/device-tool``, sending the
query rather than notes. The box embeds only its own notes and keeps no Mac index copy.
When the Mac is disconnected, recall is off or the Mac does not answer, recall says so plainly.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt
from tools.registry import tool_error, tool_result

from . import notes as notes_mod
from .index import NoteIndex

logger = logging.getLogger(__name__)

PROVIDER_NAME = "iollo_notes"
NOTES_DIRNAME = "memory-notes"
INDEX_FILENAME = "memory-index.db"
PROFILE_CHAR_CAP = 16_000
PREFETCH_KEEP = 3
SEARCH_CANDIDATES = 10
ROUTE_CANDIDATES = 50
# Local filing: attach to the best hit only at this normalised score (see index.py): a note
# ranked first by both FTS and vectors, or an alias match plus a top FTS hit.
ROUTE_MIN_SCORE = 0.9
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
NO_PROFILE_LINE = "USER PROFILE: no profile yet."
_PROFILE_SEP = "═" * 46
_AUTO = object()


MEMORY_SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": (
        "Search the user's memory notes (people, organizations, preferences, past conversations, projects, "
        "documents). "
        "Only the profile is in your prompt; everything else is found here. Search BEFORE saying you "
        "don't know or don't remember something about the user, and before asking them to repeat it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words (a name, a topic, a question)."},
            "type": {"type": "string", "enum": list(notes_mod.NOTE_TYPES), "description": "Only notes of this type."},
            "limit": {"type": "integer", "description": "Maximum hits (default 5)."},
        },
        "required": ["query"],
    },
}

MEMORY_READ_SCHEMA = {
    "name": "memory_read",
    "description": "Read one memory note in full, by the id memory_search returned (e.g. people/ana-costa). "
                   "Also lists the ids it links to.",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Note id, e.g. people/ana-costa."}},
        "required": ["id"],
    },
}

MEMORY_NOTE_SCHEMA = {
    "name": "memory_note",
    "description": (
        "Record one durable fact about the user or their world (a preference, a person, an organization, "
        "something decided in a conversation). It is filed into the right note automatically, dated and "
        "stamped with where it came from. Use this for anything worth remembering. USER.md (the profile) is "
        "generated from these notes. For owner-requested corrections/removals, first memory_read the exact note, "
        "then use correct/remove with its id, complete expected_content and exact old fact. correct replaces only "
        "that fact with replacement; remove deletes only that fact. remove_note deletes the whole selected note. "
        "Use only the scope the owner requested; preserve unrelated facts. Inspect relevant learned procedures "
        "with skill and correct/remove them with native skill_manage only when the owner's request includes them. "
        "Report the returned scope precisely: retained conversations and original sources are unchanged, and "
        "old evidence may be learned again. Never claim complete forgetting from a notes-only change. "
        "Generated USER.md is invalidated and rebuilt by normal maintenance; handwritten profiles stay intact."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["record", "correct", "remove", "remove_note"],
                       "description": "Default record. Mutations require exact id and expected_content from memory_read."},
            "id": {"type": "string", "description": "Exact note id for correction/removal."},
            "expected_content": {"type": "string", "description": "Complete unchanged content returned by memory_read; refuses stale edits."},
            "replacement": {"type": "string", "description": "New fact for correct; dated with this conversation's provenance."},
            "fact": {"type": "string", "description": "New fact for record; exact old fact text for correct/remove."},
            "type_hint": {"type": "string", "enum": list(notes_mod.NOTE_TYPES),
                          "description": "The kind of note this belongs to, if a new note is needed."},
            "about": {"type": "string", "description": "Who or what the fact is about (e.g. 'Ana')."},
            "valid_until": {"type": "string", "description": "YYYY-MM-DD after which the fact stops being true."},
        },
        "required": [],
    },
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


def _load_plugin_config() -> dict:
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        return dict(cfg_get(load_config_readonly(), "memory", PROVIDER_NAME, default={}) or {})
    except Exception:
        return {}


def _today() -> str:
    return _dt.date.today().isoformat()


class IolloNotesProvider(MemoryProvider):
    """Typed notes + local search; the profile in every turn."""

    def __init__(self, config: Optional[dict] = None, *, embedder: Any = _AUTO):
        self._config = dict(config if config is not None else _load_plugin_config())
        self._embedder = embedder  # _AUTO = the shipped model (tests pass a fake or None)
        self._index: Optional[NoteIndex] = None
        self._home: Optional[Path] = None
        self._session_id = ""
        self._write_lock = threading.Lock()
        self._last_recall = 0

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        return True  # SQLite FTS always works; vectors are optional

    # -- config ----------------------------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "device", "description": "This device in provenance: box or mac:<id>", "default": "box"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from hermes_cli.config import save_config

        save_config({"memory": {PROVIDER_NAME: dict(values)}}, merge_existing=True)

    @property
    def device(self) -> str:
        return notes_mod.one_line(str(self._config.get("device") or "box")) or "box"

    @property
    def notes_dir(self) -> Path:
        return self._require_home() / NOTES_DIRNAME

    @property
    def _idx(self) -> NoteIndex:
        if self._index is None:
            raise RuntimeError("iollo_notes is not initialized")
        return self._index

    def _require_home(self) -> Path:
        if self._home is None:
            from hermes_constants import get_hermes_home

            self._home = Path(get_hermes_home())
        return self._home

    # -- lifecycle -------------------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        from .embedder import load_embedder

        home = kwargs.get("hermes_home")
        if home:
            self._home = Path(home)
        home = self._require_home()
        self._session_id = session_id or ""
        embedder = load_embedder() if self._embedder is _AUTO else self._embedder
        self._index = NoteIndex(home / NOTES_DIRNAME, home / INDEX_FILENAME, embedder)
        try:
            self._idx.refresh()
        except Exception as exc:
            logger.warning("iollo_notes index refresh failed at start: %s", type(exc).__name__)

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id or self._session_id

    def shutdown(self) -> None:
        if self._index is not None:
            try:
                self._index.close()
            except Exception:
                pass
        self._index = None

    def backup_paths(self) -> List[str]:
        return []  # notes and index both live inside HERMES_HOME

    # -- the profile -----------------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        path = self.notes_dir / notes_mod.PROFILE_NAME
        try:
            text = path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError):
            text = ""
        if not text:
            return NO_PROFILE_LINE
        if len(text) > PROFILE_CHAR_CAP:
            cut = text[:PROFILE_CHAR_CAP]
            text = cut[: cut.rfind("\n")] if "\n" in cut else cut
        from tools.threat_patterns import scan_for_threats

        findings = scan_for_threats(text, scope="strict")
        if findings:
            logger.warning("iollo_notes USER.md blocked at load time: %s", ", ".join(findings))
            text = (f"[BLOCKED: USER.md contained threat pattern(s): {', '.join(findings)}. "
                    "Removed from the system prompt until the box regenerates it.]")
        from tools.memory_tool_store import MEMORY_BLOCK_HEADERS

        return f"{_PROFILE_SEP}\n{MEMORY_BLOCK_HEADERS['user']}\n{_PROFILE_SEP}\n{text}"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall = 0
        if self._index is None or is_trivial_prompt(query):
            return ""
        try:
            self._idx.refresh()
            hits = self._idx.search(query, limit=SEARCH_CANDIDATES)
        except Exception as exc:
            logger.warning("iollo_notes prefetch search failed: %s", type(exc).__name__)
            return ""
        if not hits:
            return ""
        kept = hits[:PREFETCH_KEEP]
        self._last_recall = len(kept)
        return "## Memory notes (use memory_read for the whole note)\n" + "\n".join(
            f"- {h['id']} ({h['type']}): {h['title']}" + "".join(f"\n  - {b}" for b in h["bullets"]) for h in kept
        )

    def recall_status(self) -> Optional[RecallStatus]:
        return RecallStatus("Iollo notes", self._last_recall) if self._last_recall else None

    # -- tools -----------------------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MEMORY_SEARCH_SCHEMA, MEMORY_READ_SCHEMA, MEMORY_NOTE_SCHEMA, RECALL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        handler = {"memory_search": self._tool_search, "memory_read": self._tool_read,
                   "memory_note": self._tool_note, "recall": self._tool_recall}.get(tool_name)
        if handler is None:
            return tool_error(f"Unknown tool: {tool_name}")
        if self._index is None:
            return tool_error("iollo_notes is not initialized")
        try:
            return handler(args or {})
        except (KeyError, ValueError, notes_mod.NoteError) as exc:
            return tool_error(str(exc))
        except Exception as exc:
            logger.warning("iollo_notes %s failed: %s", tool_name, type(exc).__name__)
            return tool_error(f"{tool_name} failed: {type(exc).__name__}")

    def _tool_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        note_type = args.get("type") or None
        if note_type and note_type not in notes_mod.TYPE_FOLDERS:
            raise ValueError(f"type must be one of {', '.join(notes_mod.NOTE_TYPES)}")
        try:
            limit = max(1, min(int(args.get("limit") or 5), 20))
        except (TypeError, ValueError):
            limit = 5
        self._idx.refresh()
        hits = self._idx.search(query, note_type=note_type, limit=limit)
        return tool_result({"results": hits, "count": len(hits)})

    def _tool_read(self, args: Dict[str, Any]) -> str:
        note_id = str(args.get("id") or "").strip().removesuffix(".md")
        if notes_mod.path_errors(notes_mod.note_path(note_id)):
            raise ValueError("not a note id (expected e.g. people/ana-costa)")
        path = self.notes_dir / notes_mod.note_path(note_id)
        if not path.is_file() or path.is_symlink():
            return tool_error(f"No note {note_id}")
        text = path.read_text(encoding="utf-8")
        return tool_result({"id": note_id, "content": text, "links": notes_mod.links(text)})

    def _tool_note(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action") or "record")
        if action != "record":
            return tool_result(self._change_note(action, args))
        return tool_result(self.record_fact(
            str(args.get("fact") or ""), type_hint=args.get("type_hint") or None,
            about=args.get("about") or None, valid_until=args.get("valid_until") or None,
        ))

    def _change_note(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Owner-scoped, optimistic edits of one retained note; never an all-store forget."""
        from tools.memory_tool_store import MemoryStore, _scan_memory_content
        from tools.skill_provenance import is_unattended_review

        if is_unattended_review():
            raise ValueError("Unattended review cannot correct or remove retained notes; use an owner-directed turn")
        if action not in {"correct", "remove", "remove_note"}:
            raise ValueError("Unknown memory_note action")
        note_id = str(args.get("id") or "")
        if notes_mod.path_errors(notes_mod.note_path(note_id)):
            raise ValueError("A valid exact note id is required")
        expected = args.get("expected_content")
        if not isinstance(expected, str) or not expected:
            raise ValueError("Read the note first and supply its complete expected_content")
        fact = str(args.get("fact") or "")
        replacement = notes_mod.one_line(str(args.get("replacement") or ""))
        if action == "correct":
            if not replacement:
                raise ValueError("replacement is required")
            if error := _scan_memory_content(replacement):
                raise ValueError(error)
        path = self.notes_dir / notes_mod.note_path(note_id)
        profile_state = "unchanged"
        with self._write_lock, MemoryStore._file_lock(self.notes_dir):
            if path.parent.is_symlink() or path.is_symlink() or not path.is_file():
                raise ValueError("Note not found or unsafe path")
            content = path.read_text(encoding="utf-8")
            if content != expected:
                raise ValueError("Note changed; read it again before correcting/removing")
            if action == "remove_note":
                path.unlink()
            else:
                note = notes_mod.parse(content)
                matches = [item for item in note.facts if item.text == fact]
                if len(matches) != 1:
                    raise ValueError("fact must match exactly one existing fact; no changes made")
                target = matches[0]
                if action == "correct":
                    provenance = self.device + (f", conversation {self._session_id}" if self._session_id else "")
                    corrected = notes_mod.Fact(_today(), replacement, provenance=provenance)
                    note.body[note.body.index(target)] = corrected
                else:
                    note.body.remove(target)
                note.updated = _today()
                rendered = notes_mod.render(note)
                if problems := notes_mod.validate(notes_mod.note_path(note_id), rendered):
                    raise ValueError("Invalid note edit: " + "; ".join(problems))
                self._write_atomic(path, rendered)
            profile = self.notes_dir / notes_mod.PROFILE_NAME
            if profile.is_file() and not profile.is_symlink():
                head = profile.read_text(encoding="utf-8-sig")[:400]
                if head.startswith(("# Profile\nGenerated from memory notes on ",
                                    "# Profile\nCurrent focus from the owner's devices; the full profile follows the next memory pass")):
                    profile.unlink()
                    profile_state = "invalidated; normal maintenance rebuilds it"
                else:
                    profile_state = "handwritten profile retained; inspect separately if it contains this fact"
        self._idx.refresh()
        return {"success": True, "action": action, "id": note_id,
                "scope": "one note" if action == "remove_note" else "one fact in one note",
                "profile": profile_state,
                "retained": ["conversation history", "learned skills", "original sources"],
                "limitation": "Old retained evidence may be learned again; this is not complete cross-store forgetting. "
                              "Stored prompts in existing conversations may still contain earlier context."}

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
        query = notes_mod.one_line(str(args.get("query") or ""))[:500]
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
        return tool_result(out)

    # -- writing ---------------------------------------------------------------------------------

    def record_fact(self, fact: str, *, type_hint: Optional[str] = None, about: Optional[str] = None,
                    valid_until: Optional[str] = None, provenance: Optional[str] = None) -> Dict[str, Any]:
        """Append a fact to a strong local match or a new typed note."""
        fact = notes_mod.one_line(fact)
        about = notes_mod.one_line(about or "") or None
        if not fact:
            raise ValueError("fact is required")
        if type_hint and type_hint not in notes_mod.TYPE_FOLDERS:
            raise ValueError(f"type_hint must be one of {', '.join(notes_mod.NOTE_TYPES)}")
        from tools.memory_tool_store import _scan_memory_content

        if scan_error := _scan_memory_content(f"{about or ''} {fact}"):
            raise ValueError(scan_error)
        if valid_until:
            _dt.date.fromisoformat(valid_until)  # ValueError on a bad date
        self._idx.refresh()
        query = f"{about} {fact}" if about else fact
        hits = self._idx.search(query, limit=ROUTE_CANDIDATES)
        best = hits[0] if hits else None
        if best and best["score"] >= ROUTE_MIN_SCORE and (not type_hint or best["type"] == type_hint):
            target_id, new_type = best["id"], None
        else:
            target_id, new_type = None, type_hint or "preference"
        if provenance is None:
            provenance = self.device + (f", conversation {self._session_id}" if self._session_id else "")
        note_id, created = self._append(target_id, new_type, fact, about, valid_until, provenance)
        self._idx.refresh()
        return {"success": True, "id": note_id, "created": created}

    def _append(self, target_id: Optional[str], new_type: Optional[str], fact: str, about: Optional[str],
                valid_until: Optional[str], provenance: str):
        from tools.memory_tool_store import MemoryStore

        today = _today()
        root = self.notes_dir
        with self._write_lock, MemoryStore._file_lock(root):  # lock file: $HERMES_HOME/memory-notes.lock
            created = False
            if target_id is None:
                assert new_type is not None
                title = about or " ".join(fact.split()[:8]).rstrip(".,;:")
                target_id = f"{notes_mod.TYPE_FOLDERS[new_type]}/{notes_mod.slugify(title)}"
            path = root / notes_mod.note_path(target_id)
            if path.is_file():
                note = notes_mod.parse(path.read_text(encoding="utf-8"))
            else:
                kind = notes_mod.FOLDER_TYPES[target_id.split("/", 1)[0]]
                title = about or " ".join(fact.split()[:8]).rstrip(".,;:")
                note = notes_mod.new_note(target_id, kind, title, [about] if about and about != title else [], today)
                created = True
            notes_mod.append_fact(note, fact, today, provenance, valid_until)
            text = notes_mod.render(note)
            if problems := notes_mod.validate(notes_mod.note_path(target_id), text):
                raise ValueError(f"note {target_id} would be invalid ({'; '.join(problems)}); "
                                 "the box consolidates full notes, try again later")
            self._write_atomic(path, text)
        return target_id, created

    def _write_atomic(self, path: Path, text: str) -> None:
        """Temp file outside the git folder, fsync, then rename into place with mode 0644."""
        tmp_dir = self._require_home() / (NOTES_DIRNAME + ".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=tmp_dir, suffix=".md")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- legacy memory tool ----------------------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """A built-in ``memory`` tool write (older profiles) becomes a memory_note."""
        if action not in ("add", "replace") or not (content or "").strip() or self._index is None:
            return
        try:
            self.record_fact(content, type_hint="preference" if target == "user" else "conversation")
        except Exception as exc:
            logger.debug("iollo_notes legacy memory write not recorded: %s", type(exc).__name__)


def register(ctx) -> None:
    """Register the iollo_notes memory provider with the plugin system."""
    ctx.register_memory_provider(IolloNotesProvider())
