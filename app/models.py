"""Pydantic schemas for Phase 1 (ingestion/chunking) and beyond."""
from __future__ import annotations

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """One chunk of text extracted from an uploaded document, ready to be
    embedded and stored. This is the RAG equivalent of the YT project's
    TranscriptChunk - but notice there's no start_time/end_time here,
    because there's no timeline to anchor to. Instead we keep word-offset
    and page info, which is what lets us cite "page 4" later instead of
    "12:34" like the YT project could."""
    chunk_id: str          # unique, e.g. "{doc_id}::chunk::{index}"
    doc_id: str
    doc_name: str
    chunk_index: int
    text: str
    page_number: int | None = None   # only meaningful for PDFs
    content_hash: str | None = None  # SHA256 of the source file, for dedup
    stored_filename: str | None = None  # actual filename on disk in uploads/, for cleanup on delete


class UploadResponse(BaseModel):
    doc_id: str
    doc_name: str
    num_chunks: int
    num_pages: int | None = None


class ChatMessage(BaseModel):
    """One turn in a conversation. role is 'user' or 'assistant'. sources
    is only populated on assistant messages (the chunks that answer was
    grounded in) - kept here rather than as a separate lookup so a
    session's full history, sources included, is one self-contained object."""
    role: str
    content: str
    sources: list[dict] | None = None


class ChatSession(BaseModel):
    """A running conversation. Deliberately just an ordered list of
    messages - no separate 'current topic' or state tracking. Everything
    Phase 6's query rewriting needs (previous questions, previous answers)
    can be derived from this list alone."""
    session_id: str
    messages: list[ChatMessage] = []