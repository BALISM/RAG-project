"""
Chat session storage, persisted to SQLite.

Each session stores its full message history as a JSON blob.  Sessions now
include metadata (title, timestamps, message count) to support the
multi-session sidebar UI.

The public API has the same core signatures as the original in-memory
version, so the upgrade was transparent to main.py and rag.py.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime

from app.config import settings
from app.logging_config import get_logger
from app.models import ChatMessage, ChatSession, SessionSummary

logger = get_logger(__name__)

_connection: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(
            str(settings.sessions_db_path),
            check_same_thread=False,
        )
        _connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                title        TEXT NOT NULL DEFAULT 'New Chat',
                data         TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                last_active  TEXT NOT NULL
            )
        """)
        _connection.commit()
    return _connection


def _save(session: ChatSession) -> None:
    conn = _get_connection()
    conn.execute(
        """INSERT INTO sessions (session_id, title, data, created_at, last_active)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               title = excluded.title,
               data = excluded.data,
               last_active = excluded.last_active""",
        (
            session.session_id,
            session.title,
            session.model_dump_json(),
            session.created_at,
            session.last_active,
        ),
    )
    conn.commit()


# ─── CRUD ─────────────────────────────────────────────────────────────────────


def create_session() -> ChatSession:
    """Create a fresh, empty chat session."""
    now = datetime.now().isoformat()
    session = ChatSession(
        session_id=uuid.uuid4().hex[:12],
        title="New Chat",
        created_at=now,
        last_active=now,
    )
    _save(session)
    logger.info("Created session %s", session.session_id)
    return session


def get_session(session_id: str) -> ChatSession | None:
    """Retrieve a session by ID, or None if not found."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT data FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return ChatSession.model_validate_json(row[0])


def get_or_create_session(session_id: str | None) -> ChatSession:
    """If the caller passed a known session_id, reuse it; otherwise create."""
    if session_id:
        existing = get_session(session_id)
        if existing:
            return existing
    return create_session()


def append_message(session_id: str, message: ChatMessage) -> None:
    """Add a message to a session and update metadata."""
    session = get_session(session_id)
    if session is None:
        raise KeyError(f"No session with id {session_id}")

    session.messages.append(message)
    session.last_active = datetime.now().isoformat()

    # Auto-generate title from the first user question
    if session.title == "New Chat" and message.role == "user":
        title = message.content.strip()[:80]
        if len(message.content.strip()) > 80:
            title += "…"
        session.title = title

    _save(session)


def get_history(session_id: str) -> list[ChatMessage]:
    """Return the message list for a session, or empty if not found."""
    session = get_session(session_id)
    return session.messages if session else []


# ─── Session Management ──────────────────────────────────────────────────────


def list_sessions() -> list[SessionSummary]:
    """Return all sessions, most recently active first."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT session_id, title, data, created_at, last_active "
        "FROM sessions ORDER BY last_active DESC"
    ).fetchall()

    summaries = []
    for session_id, title, data, created_at, last_active in rows:
        try:
            session = ChatSession.model_validate_json(data)
            msg_count = len(session.messages)
        except Exception:
            msg_count = 0

        summaries.append(
            SessionSummary(
                session_id=session_id,
                title=title or "New Chat",
                message_count=msg_count,
                created_at=created_at or "",
                last_active=last_active or "",
            )
        )
    return summaries


def delete_session(session_id: str) -> bool:
    """Delete a session.  Returns True if it existed, False otherwise."""
    conn = _get_connection()
    cursor = conn.execute(
        "DELETE FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    conn.commit()
    deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Deleted session %s", session_id)
    return deleted


def get_session_count() -> int:
    """Return the total number of stored sessions."""
    conn = _get_connection()
    row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    return row[0] if row else 0