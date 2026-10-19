"""iollo_notes recall (iollo docs/MAC-RECALL-AND-MEMORY-DESIGN.md sections 6 and 7): the ``recall`` tool against a stub control plane (the Mac is
the only place recall searches; iollo owner decision 2026-10-15).
Every name, ref and token here is fake."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from plugins.memory.iollo_notes import IolloNotesProvider, where_it_is


# -- where a hit is ------------------------------------------------------------------------------

def test_where_it_is_reads_like_a_place():
    assert where_it_is({"source": "files", "kind": "pdf", "ref": "Documents/Taxes/2025/irs.pdf", "loc": 3}) == \
        "Documents/Taxes/2025, irs.pdf, page 3"
    assert where_it_is({"source": "files", "kind": "audio", "ref": "Desktop/call.m4a", "loc": 83}) == \
        "Desktop, call.m4a, at 1:23"
    assert where_it_is({"source": "messages", "people": ["Fake Ana"], "when": "2026-03-03T10:00:00Z"}) == \
        "message from Fake Ana on 2026-03-03"
    assert where_it_is({"source": "mail", "people": ["Fake Ana"], "when": "2026-03-03", "title": "Lease"}) == \
        'email from Fake Ana on 2026-03-03 ("Lease")'
    assert where_it_is({"source": "notes", "title": "Porto"}) == 'note "Porto" in Notes'
    assert where_it_is({"source": "history", "ref": "https://www.example.com/a", "when": "2026-10-01"}) == \
        "visited www.example.com on 2026-10-01"


# -- the recall tool -----------------------------------------------------------------------------

class Stub:
    """One HTTP server playing the control plane (/relay/device-tool); anything else it records and answers 404."""

    def __init__(self):
        self.requests, self.listing, self.mac_answer = [], {"connected": False, "linked": True}, None
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, payload, status=200):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                stub.requests.append(("GET", self.path, None, dict(self.headers)))
                self._answer(stub.listing)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append(("POST", self.path, body, dict(self.headers)))
                if self.path == "/relay/device-tool":
                    self._answer(stub.mac_answer)
                else:
                    self._answer({"detail": "Not Found"}, status=404)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub(monkeypatch):
    s = Stub()
    monkeypatch.setenv("RELAY_URL", s.base)
    monkeypatch.setenv("RELAY_TOKEN", "fake-relay-token")
    yield s
    s.close()


@pytest.fixture
def provider(tmp_path, stub):
    p = IolloNotesProvider()
    p.initialize("sess-fake", hermes_home=str(tmp_path), platform="cli")
    yield p
    p.shutdown()


MAC_HIT = {"ref": "Documents/Housing/porto-lease.pdf", "source": "files", "kind": "pdf", "title": "Porto lease",
           "when": "2026-09-01T00:00:00Z", "people": ["Fake Landlord"], "excerpt": "Rent due on the first.",
           "score": 1.0}
MAC_LISTING = {"connected": True, "device": {"id": "dev_fake_mac", "name": "Fake Mac"},
               "tools": [{"name": "mac_recall_local"}], "recall": {"enabled": True, "items": 10, "indexed": 10,
                                                                    "last_pass_at": None}}


def test_recall_is_a_tool_of_the_provider(provider):
    names = [schema["name"] for schema in provider.get_tool_schemas()]
    assert names == ["recall"]


def test_recall_asks_only_the_connected_mac(provider, stub):
    stub.listing = {**MAC_LISTING, "recall": {"enabled": True, "items": 10, "indexed": 4, "last_pass_at": None}}
    stub.mac_answer = {"state": "done", "output": "2 hits", "error": None, "data": {
        "hits": [{**MAC_HIT, "loc": 3, "item": 7}, {"source": "messages", "kind": "message", "ref": "msg:1",
                                                    "people": ["Fake Landlord"], "when": "2026-10-10T09:00:00Z",
                                                    "excerpt": "The keys are with me."},
                 {**MAC_HIT, "loc": 4}],
        "indexed": 4, "total": 10}}
    out = json.loads(provider.handle_tool_call("recall", {"query": "porto lease", "sources": ["files"],
                                                          "since": "2026-01-01", "limit": 50}))
    assert out["mac"] == "searched" and [h["from"] for h in out["hits"]] == ["mac", "mac"]   # one hit per item
    assert out["hits"][0]["where"] == "Documents/Housing, porto-lease.pdf, page 3"
    assert out["hits"][1]["where"] == "message from Fake Landlord on 2026-10-10"
    assert out["still_reading"] is True and "still reading your Mac" in out["note"]
    call = next(r for r in stub.requests if r[0] == "POST" and r[1] == "/relay/device-tool")
    assert call[2] == {"tool": "mac_recall_local", "args": {"query": "porto lease", "limit": 20, "sources": ["files"],
                                                             "since": "2026-01-01"},
                       "deadline_s": 20, "device_id": "dev_fake_mac"}
    assert call[3]["X-Relay-Token"] == "fake-relay-token"
    assert [r[1] for r in stub.requests] == ["/relay/device-tool", "/relay/device-tool"]   # no box catalogue search


def test_zero_hit_recall_logs_a_miss_without_the_query(provider, stub, caplog):
    stub.listing = MAC_LISTING
    stub.mac_answer = {"state": "done", "output": "", "error": None, "data": {"hits": [], "indexed": 10, "total": 10}}
    with caplog.at_level("INFO", logger="plugins.memory.iollo_notes"):
        provider.handle_tool_call("recall", {"query": "secret porto lease"})
    misses = [r.getMessage() for r in caplog.records if r.getMessage().startswith("recall miss")]
    assert misses == ["recall miss: state=searched query_chars=18 sources=0"]
    assert "secret" not in misses[0]


def test_recall_says_nothing_found_only_after_the_mac_searched(provider, stub):
    stub.listing = MAC_LISTING
    stub.mac_answer = {"state": "done", "output": "", "error": None, "data": {"hits": [], "indexed": 10, "total": 10}}
    out = json.loads(provider.handle_tool_call("recall", {"query": "porto lease"}))
    assert out["mac"] == "searched" and out["hits"] == [] and out["note"].startswith("Nothing found")
    stub.listing = {**MAC_LISTING, "recall": {"enabled": True, "items": 5, "indexed": 0}}
    stub.mac_answer = {"state": "failed", "output": "", "data": {"hits": [MAC_HIT], "indexed": 0, "total": 5},
                       "error": {"code": "indexing", "message": "first pass", "retryable": True}}
    out = json.loads(provider.handle_tool_call("recall", {"query": "porto lease"}))
    assert out["mac"] == "indexing" and out["hits"][0]["from"] == "mac" and out["still_reading"] is True
    assert "error" in json.loads(provider.handle_tool_call("recall", {"query": "  "}))


def test_recall_says_plainly_when_the_mac_cannot_be_searched(provider, stub, monkeypatch):
    out = json.loads(provider.handle_tool_call("recall", {"query": "porto lease"}))          # not connected
    assert out == {"error": out["error"], "mac": "not_connected"} and "not connected" in out["error"]
    stub.listing = {**MAC_LISTING, "recall": {"enabled": False, "items": 0, "indexed": 0}}
    out = json.loads(provider.handle_tool_call("recall", {"query": "porto lease"}))
    assert out["mac"] == "recall_off" and "Recall is off" in out["error"] and "hits" not in out
    stub.listing = MAC_LISTING
    stub.mac_answer = {"state": "failed", "output": "", "data": {},
                       "error": {"code": "mac_unreachable", "message": "asleep", "retryable": True}}
    out = json.loads(provider.handle_tool_call("recall", {"query": "porto lease"}))
    assert out["mac"] == "unreachable" and "did not answer" in out["error"] and "hits" not in out
    monkeypatch.setenv("RELAY_URL", "http://127.0.0.1:9")    # nothing listens: the control plane is unreachable
    out = json.loads(provider.handle_tool_call("recall", {"query": "porto lease"}))
    assert out["mac"] == "unreachable" and "hits" not in out


def test_recall_without_a_relay_token_says_the_mac_is_not_connected(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAY_TOKEN", raising=False)
    p = IolloNotesProvider()
    p.initialize("sess-fake", hermes_home=str(tmp_path), platform="cli")
    out = json.loads(p.handle_tool_call("recall", {"query": "porto lease"}))
    assert out["mac"] == "not_connected" and "hits" not in out and "error" in out
    p.shutdown()
