"""
Phase 6a — Session storage.

A simple in-memory dict, same pattern as the YT project's job store: fine
for a single-process demo, and swappable for Redis/a database later without
the calling code (main.py, rag.py) needing to change - they only ever call
these functions, never touch the dict directly.

Deliberately NOT persisted to disk. Chat history disappearing on a server
restart is a much smaller deal than the vector store disappearing (which
would mean re-uploading and re-embedding every document) - so this gets to
stay simple.
"""
from __future__ import annotations

import uuid

from app.models import ChatMessage, ChatSession

_SESSIONS: dict[str, ChatSession] = {}


def create_session() -> ChatSession:
    session_id = uuid.uuid4().hex[:12]
    session = ChatSession(session_id=session_id)
    _SESSIONS[session_id] = session
    return session


def get_session(session_id: str) -> ChatSession | None:
    return _SESSIONS.get(session_id)


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
    session = _SESSIONS.get(session_id)
    if session is None:
        raise KeyError(f"No session with id {session_id}")
    session.messages.append(message)


def get_history(session_id: str) -> list[ChatMessage]:
    session = _SESSIONS.get(session_id)
    return session.messages if session else []