"""Pydantic schemas for every data structure in the application.

Models are grouped by domain: documents/chunks, chat, API responses,
and system status.  Every model has field descriptions so FastAPI's
auto-generated OpenAPI docs are genuinely useful.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ── Document & Chunk Models ──────────────────────────────────────────────────


class DocumentChunk(BaseModel):
    """One chunk of text extracted from an uploaded document, ready to be
    embedded and stored."""

    chunk_id: str = Field(description="Unique chunk identifier, e.g. '{doc_id}::chunk::{index}'")
    doc_id: str = Field(description="Parent document identifier")
    doc_name: str = Field(description="Original filename as uploaded")
    chunk_index: int = Field(description="Zero-based position within the document")
    text: str = Field(description="The chunk's text content")
    page_number: int | None = Field(default=None, description="Source page (PDF only)")
    content_hash: str | None = Field(default=None, description="SHA-256 of the source file for dedup")
    stored_filename: str | None = Field(default=None, description="Actual filename on disk in uploads/")


class DocumentMetadata(BaseModel):
    """Summary info about a stored document, returned by list/detail endpoints."""

    doc_id: str
    doc_name: str
    num_chunks: int
    uploaded_at: str | None = Field(default=None, description="ISO-8601 upload timestamp")
    file_size_bytes: int | None = Field(default=None, description="Original file size")
    word_count: int | None = Field(default=None, description="Total word count across all chunks")


class UploadResponse(BaseModel):
    """Returned after a successful document upload."""

    doc_id: str
    doc_name: str
    num_chunks: int
    num_pages: int | None = None
    duplicate: bool = Field(default=False, description="True if this file was already uploaded (reused)")


# ── Search Models ────────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    """A single search hit with relevance score."""

    chunk_id: str
    doc_id: str
    doc_name: str
    chunk_index: int
    text: str
    page_number: int | None = None
    relevance_score: float = Field(description="Cosine similarity score (0–1, higher = more relevant)")
    excerpt: str = Field(description="First 200 characters of the chunk text")


# ── Chat Models ──────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """One turn in a conversation."""

    role: str = Field(description="Either 'user' or 'assistant'")
    content: str = Field(description="Message text")
    sources: list[dict] | None = Field(default=None, description="Source citations (assistant only)")


class ChatSession(BaseModel):
    """A running conversation with metadata."""

    session_id: str
    title: str = Field(default="New Chat", description="Auto-generated from the first question")
    messages: list[ChatMessage] = []
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = Field(default_factory=lambda: datetime.now().isoformat())


class SessionSummary(BaseModel):
    """Lightweight session info for the sidebar list."""

    session_id: str
    title: str
    message_count: int
    created_at: str
    last_active: str


class ChatRequest(BaseModel):
    """Request body for /chat and /chat/stream."""

    question: str = Field(min_length=1, max_length=4000, description="The question to ask")
    session_id: str | None = Field(default=None, description="Omit on first message; reuse afterward")
    doc_id: str | None = Field(default=None, description="Restrict search to one document")
    doc_ids: list[str] | None = Field(default=None, description="Restrict search to these documents")


class ChatResponse(BaseModel):
    """Response body for /chat (non-streaming)."""

    session_id: str
    answer: str
    sources: list[dict]
    grounded: bool
    warning: str | None = None


# ── System Models ────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response for /health."""

    status: str = "ok"
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    """System-wide statistics."""

    total_documents: int
    total_chunks: int
    total_sessions: int
    storage_path: str


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str = Field(description="Machine-readable error type")
    detail: str = Field(description="Human-readable error message")
    status_code: int