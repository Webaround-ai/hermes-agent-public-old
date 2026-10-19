"""iollo_notes is recall-only (owner decision 2026-10-19): Hermes native memory is the only memory. The provider
exposes only ``recall``, adds nothing to the prompt, prefetches nothing and never creates a notes folder."""

from plugins.memory.iollo_notes import IolloNotesProvider


def test_provider_exposes_only_recall(tmp_path):
    p = IolloNotesProvider()
    p.initialize("sess", hermes_home=str(tmp_path))
    assert [s["name"] for s in p.get_tool_schemas()] == ["recall"]
    assert "error" in p.handle_tool_call("memory_note", {"fact": "x"})
    assert "error" in p.handle_tool_call("memory_search", {"query": "x"})


def test_prompt_block_and_prefetch_are_empty(tmp_path):
    (tmp_path / "memory-notes").mkdir()
    (tmp_path / "memory-notes" / "USER.md").write_text("Old profile", encoding="utf-8")
    p = IolloNotesProvider()
    p.initialize("sess", hermes_home=str(tmp_path))
    assert p.system_prompt_block() == ""
    assert p.prefetch("anything", session_id="sess") == ""


def test_builtin_memory_write_creates_no_notes_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.memory_tool_store import MemoryStore

    p = IolloNotesProvider()
    p.initialize("sess", hermes_home=str(tmp_path))
    store = MemoryStore()
    store.load_from_disk()
    assert store.add("user", "Prefers short answers.").get("success")
    p.on_memory_write("add", "user", "Prefers short answers.")
    assert not (tmp_path / "memory-notes").exists()
    assert not (tmp_path / "memory-notes.lock").exists()


def test_module_has_no_top_level_notes_index_or_embedder_import():
    import plugins.memory.iollo_notes as mod

    assert not hasattr(mod, "notes_mod") and not hasattr(mod, "NoteIndex")


def test_memory_full_refusal_is_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.memory_tool_store import MemoryStore

    store = MemoryStore(memory_char_limit=40, user_char_limit=40)
    store.load_from_disk()
    with caplog.at_level("INFO", logger="tools.memory_tool"):
        result = store.add("memory", "x" * 60)
    assert not result.get("success")
    assert any(r.getMessage().startswith("memory full: target=memory chars=60 limit=40") for r in caplog.records)
