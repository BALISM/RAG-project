"""
Formal test suite for chat_memory.py. The most important test here is
test_history_survives_a_simulated_restart - that's the entire reason this
module was rewritten from an in-memory dict to SQLite, so it needs to be
proven, not just assumed from "it uses a database now."
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import app.chat_memory as memory
from app.config import settings
from app.models import ChatMessage


@pytest.fixture
def isolated_sessions_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "sessions_db", str(tmp_path / "test_sessions.db"))
    monkeypatch.setattr(memory, "_connection", None)
    yield
    monkeypatch.setattr(memory, "_connection", None)


def test_create_session_gives_a_fresh_empty_session(isolated_sessions_db):
    session = memory.create_session()
    assert session.messages == []
    assert len(session.session_id) == 12


def test_get_session_returns_none_for_unknown_id(isolated_sessions_db):
    assert memory.get_session("nonexistent") is None


def test_append_and_get_history_round_trips_in_order(isolated_sessions_db):
    session = memory.create_session()
    memory.append_message(session.session_id, ChatMessage(role="user", content="q1"))
    memory.append_message(session.session_id, ChatMessage(role="assistant", content="a1"))
    memory.append_message(session.session_id, ChatMessage(role="user", content="q2"))

    history = memory.get_history(session.session_id)
    assert [m.content for m in history] == ["q1", "a1", "q2"]


def test_append_message_preserves_sources(isolated_sessions_db):
    session = memory.create_session()
    memory.append_message(
        session.session_id,
        ChatMessage(role="assistant", content="answer", sources=[{"doc_name": "cv.pdf", "page_number": 1}]),
    )
    history = memory.get_history(session.session_id)
    assert history[0].sources == [{"doc_name": "cv.pdf", "page_number": 1}]


def test_append_message_to_unknown_session_raises(isolated_sessions_db):
    with pytest.raises(KeyError):
        memory.append_message("totally-fake-id", ChatMessage(role="user", content="x"))


def test_get_or_create_session_reuses_known_id(isolated_sessions_db):
    session = memory.create_session()
    memory.append_message(session.session_id, ChatMessage(role="user", content="hello"))

    reused = memory.get_or_create_session(session.session_id)
    assert reused.session_id == session.session_id
    assert len(reused.messages) == 1


def test_get_or_create_session_falls_back_to_new_for_unknown_id(isolated_sessions_db):
    fresh = memory.get_or_create_session("this-id-was-never-created")
    assert fresh.session_id != "this-id-was-never-created"
    assert fresh.messages == []


def test_get_history_for_unknown_session_returns_empty_list_not_none(isolated_sessions_db):
    assert memory.get_history("nonexistent") == []


def test_history_survives_a_simulated_restart(isolated_sessions_db):
    """This is the entire point of the SQLite rewrite. Simulate a server
    restart by dropping the cached connection (exactly what happens when
    the process exits) and reconnecting fresh - the data should still be
    there because it was written to a file on disk, not held in memory."""
    session = memory.create_session()
    memory.append_message(session.session_id, ChatMessage(role="user", content="what is the capital of France?"))
    memory.append_message(session.session_id, ChatMessage(role="assistant", content="Paris."))

    # Simulate the process restarting: the module-level connection is gone.
    memory._connection = None

    # A fresh "process" reads from the same on-disk file...
    recovered_history = memory.get_history(session.session_id)
    assert [m.content for m in recovered_history] == [
        "what is the capital of France?",
        "Paris.",
    ], "conversation history should survive a restart since it's now persisted to disk, not held in a dict"


def test_multiple_independent_sessions_dont_interfere(isolated_sessions_db):
    s1 = memory.create_session()
    s2 = memory.create_session()
    memory.append_message(s1.session_id, ChatMessage(role="user", content="session 1 message"))
    memory.append_message(s2.session_id, ChatMessage(role="user", content="session 2 message"))

    assert [m.content for m in memory.get_history(s1.session_id)] == ["session 1 message"]
    assert [m.content for m in memory.get_history(s2.session_id)] == ["session 2 message"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))