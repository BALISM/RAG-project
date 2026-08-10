"""
Chat session storage, persisted to SQLite.

Each session stores its full message history as a JSON blob.  Sessions now
include metadata (title, timestamps, message count) to support the
multi-session sidebar UI.

Enhanced with:
  - Session renaming (PATCH endpoint support)
  - Session export in Markdown or JSON format
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime

from app.config import settings
from app.logging_config import get_logger
from app.models import ChatMessage, ChatSession, ExportFormat, SessionSummary

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


# ─── New: Session Rename ─────────────────────────────────────────────────────


def rename_session(session_id: str, new_title: str) -> bool:
    """Rename a session's title. Returns True if found and renamed."""
    session = get_session(session_id)
    if session is None:
        return False

    session.title = new_title.strip()[:120] or "New Chat"
    session.last_active = datetime.now().isoformat()
    _save(session)
    logger.info("Renamed session %s → '%s'", session_id, session.title)
    return True


# ─── New: Session Export ─────────────────────────────────────────────────────


def export_session(session_id: str, fmt: ExportFormat = ExportFormat.MARKDOWN) -> str | None:
    """Export a session in the requested format. Returns the formatted string,
    or None if the session doesn't exist."""
    session = get_session(session_id)
    if session is None:
        return None

    if fmt == ExportFormat.JSON:
        return session.model_dump_json(indent=2)

    # Markdown export
    lines = [
        f"# {session.title}",
        f"",
        f"*Session ID:* `{session.session_id}`  ",
        f"*Created:* {session.created_at}  ",
        f"*Last Active:* {session.last_active}  ",
        f"*Messages:* {len(session.messages)}",
        "",
        "---",
        "",
    ]

    for msg in session.messages:
        role_label = "🧑 **You**" if msg.role == "user" else "🤖 **AetherMind**"
        lines.append(f"### {role_label}")
        lines.append("")
        lines.append(msg.content)
        lines.append("")

        if msg.sources:
            lines.append("**Sources:**")
            for src in msg.sources:
                page_info = f", page {src.get('page_number')}" if src.get("page_number") else ""
                lines.append(f"- [{src.get('doc_name', 'Unknown')}{page_info}] — _{src.get('excerpt', '')[:100]}..._")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)