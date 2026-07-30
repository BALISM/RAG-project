"""
Phase 6 (revisited) — Session storage, now persisted to SQLite instead of
an in-memory dict, so conversation history survives a server restart.
Uses Python's built-in sqlite3 - no new dependency - storing each session
as one JSON blob per row rather than a fully relational schema, since
ChatSession is already exactly the shape we want to read/write in one
piece, and conversations are small enough that "load the whole thing,
re-save the whole thing" is simpler than tracking individual message rows.

The public functions below (create_session, get_session, etc.) have the
exact same signatures as the original in-memory version, so nothing in
main.py or rag.py needed to change when this got swapped in.
"""
from __future__ import annotations

import sqlite3
import uuid

from app.config import settings
from app.models import ChatMessage, ChatSession

_connection: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(str(settings.sessions_db_path), check_same_thread=False)
        _connection.execute(
            "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        _connection.commit()
    return _connection


def _save(session: ChatSession) -> None:
    conn = _get_connection()
    conn.execute(
        "INSERT INTO sessions (session_id, data) VALUES (?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET data = excluded.data",
        (session.session_id, session.model_dump_json()),
    )
    conn.commit()


def create_session() -> ChatSession:
    session_id = uuid.uuid4().hex[:12]
    session = ChatSession(session_id=session_id)
    _save(session)
    return session


def get_session(session_id: str) -> ChatSession | None:
    conn = _get_connection()
    row = conn.execute("SELECT data FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    return ChatSession.model_validate_json(row[0])


def get_or_create_session(session_id: str | None) -> ChatSession:
    """Convenience for the /chat endpoint: if the caller passed a
    session_id we recognize, keep using it; otherwise start a fresh one.
    This means a client can always just pass back whatever session_id it
    got last time, with no separate 'create session' round trip needed."""
    if session_id:
        existing = get_session(session_id)
        if existing:
            return existing
    return create_session()


def append_message(session_id: str, message: ChatMessage) -> None:
    session = get_session(session_id)
    if session is None:
        raise KeyError(f"No session with id {session_id}")
    session.messages.append(message)
    _save(session)


def get_history(session_id: str) -> list[ChatMessage]:
    session = get_session(session_id)
    return session.messages if session else []