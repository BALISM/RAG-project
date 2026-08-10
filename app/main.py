"""
FastAPI application — the HTTP layer connecting everything together.

Architecture:
  - All API routes prefixed with /api/v1/ for versioning
  - Static files served at / for the browser UI
  - Structured error handling via custom exception classes
  - Security headers middleware
  - Rate limiting on costly endpoints (upload, chat)
  - Rich OpenAPI metadata for auto-generated docs

Enhanced v2.0:
  - Document preview endpoint
  - Semantic search endpoint
  - Session export & rename endpoints
  - Smart suggestions endpoint
  - Answer mode support in chat
  - Enhanced stats with richer metrics
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.chat_memory import (
    append_message,
    delete_session,
    export_session,
    get_or_create_session,
    get_session,
    get_session_count,
    list_sessions,
    rename_session,
)
from app.config import settings
from app.exceptions import IngestionError, RAGBaseError
from app.ingestion import SUPPORTED_EXTENSIONS, chunk_document, compute_file_hash, get_file_metadata
from app.logging_config import get_logger, setup_logging
from app.models import (
    AnswerMode,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ExportFormat,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StatsResponse,
    UploadResponse,
)
from app.rag import GenerationError, answer_question, answer_question_stream, answer_question_stream_async, generate_suggestions, rewrite_query, rewrite_query_async
from app.vectorstore import (
    add_chunks,
    delete_document,
    find_document_by_hash,
    get_collection_stats,
    get_document,
    get_document_preview,
    get_knowledge_base_summary,
    list_documents,
    search_with_scores,
)

# ─── Logging Setup ────────────────────────────────────────────────────────────

setup_logging(settings.log_level)
logger = get_logger(__name__)

# ─── Rate Limiter ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ─── App Creation ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AetherMind AI — RAG Platform API",
    description=(
        "Upload documents, then ask questions grounded in exactly what's in them. "
        "Features semantic search, multi-mode answers, chat history, and document exploration. "
        "Built with FastAPI, Google Gemini, and ChromaDB."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Documents", "description": "Upload, list, inspect, preview, and delete documents"},
        {"name": "Chat", "description": "Ask questions with RAG-grounded answers"},
        {"name": "Search", "description": "Standalone semantic search across the knowledge base"},
        {"name": "Sessions", "description": "Manage, rename, and export conversation sessions"},
        {"name": "System", "description": "Health checks, system statistics, and suggestions"},
    ],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ─── Global Exception Handler ─────────────────────────────────────────────────


@app.exception_handler(RAGBaseError)
async def rag_error_handler(request: Request, exc: RAGBaseError):
    """Catch all application-specific errors and return structured JSON."""
    status_code = 500
    if isinstance(exc, IngestionError):
        status_code = 422
    elif isinstance(exc, GenerationError):
        status_code = 502

    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=type(exc).__name__,
            detail=exc.detail,
            status_code=status_code,
        ).model_dump(),
    )


# ─── Startup ──────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    issues = settings.validate_startup()
    for issue in issues:
        logger.warning("⚠️  %s", issue)
    logger.info("🚀 AetherMind AI v2.0.0 started")
    logger.info("   Gemini model: %s", settings.gemini_model)
    logger.info("   Embedding model: %s", settings.embedding_model)
    logger.info("   Chunk size: %d words, overlap: %d words", settings.chunk_size_words, settings.chunk_overlap_words)
    logger.info("   Max documents: %d, max file size: %dMB", settings.max_documents, settings.max_file_size_mb)


# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENT ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@app.post(
    "/api/v1/documents/upload",
    response_model=UploadResponse,
    tags=["Documents"],
    summary="Upload a document",
    description="Upload a PDF, DOCX, TXT, or MD file. The document is extracted, chunked, embedded, and stored.",
)
@limiter.limit(settings.rate_limit_upload)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Save to disk with a random name to avoid collisions
    temp_name = f"{uuid.uuid4().hex[:12]}{suffix}"
    dest_path = settings.upload_path / temp_name
    max_bytes = settings.max_file_size_bytes

    try:
        bytes_written = 0
        with dest_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_file_size_mb}MB upload limit",
                    )
                f.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    try:
        # Check for duplicate
        file_hash = compute_file_hash(dest_path)
        existing_doc_id = find_document_by_hash(file_hash)
        if existing_doc_id:
            dest_path.unlink(missing_ok=True)
            logger.info("Duplicate upload of %s → reusing doc_id=%s", file.filename, existing_doc_id)
            existing = next((d for d in list_documents() if d["doc_id"] == existing_doc_id), None)
            return UploadResponse(
                doc_id=existing_doc_id,
                doc_name=existing["doc_name"] if existing else (file.filename or temp_name),
                num_chunks=existing["num_chunks"] if existing else 0,
                num_pages=None,
                duplicate=True,
            )

        # Check document count limit
        current_count = len(list_documents())
        if current_count >= settings.max_documents:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Storage limit reached ({settings.max_documents} documents). "
                    "Delete an existing document before uploading a new one."
                ),
            )

        # Ingest
        doc_id, chunks = chunk_document(dest_path, doc_name=file.filename or temp_name)
        add_chunks(chunks, file_size_bytes=bytes_written)

    except HTTPException:
        raise
    except IngestionError as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        logger.exception("Unexpected error ingesting %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}") from e

    page_numbers = {c.page_number for c in chunks if c.page_number is not None}
    logger.info("Ingested %s → doc_id=%s, %d chunks", file.filename, doc_id, len(chunks))

    return UploadResponse(
        doc_id=doc_id,
        doc_name=file.filename or temp_name,
        num_chunks=len(chunks),
        num_pages=max(page_numbers) if page_numbers else None,
    )


@app.get(
    "/api/v1/documents",
    tags=["Documents"],
    summary="List all documents",
    description="Returns all uploaded documents with chunk counts and metadata.",
)
def get_documents() -> list[dict]:
    return list_documents()


@app.get(
    "/api/v1/documents/{doc_id}",
    tags=["Documents"],
    summary="Get document details",
    description="Returns full detail for one document including all its chunks.",
)
def get_document_detail(doc_id: str) -> dict:
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No document found with doc_id={doc_id}")
    return doc


@app.get(
    "/api/v1/documents/{doc_id}/preview",
    tags=["Documents"],
    summary="Preview a document",
    description="Returns the first few chunks of a document for quick inline preview.",
)
def preview_document(doc_id: str, max_chunks: int = Query(default=3, ge=1, le=10)) -> dict:
    preview = get_document_preview(doc_id, max_chunks=max_chunks)
    if preview is None:
        raise HTTPException(status_code=404, detail=f"No document found with doc_id={doc_id}")
    return preview


@app.delete(
    "/api/v1/documents/{doc_id}",
    tags=["Documents"],
    summary="Delete a document",
    description="Removes a document's vectors from the store and deletes the file from disk.",
)
def remove_document(doc_id: str) -> dict:
    delete_document(doc_id)
    return {"deleted": doc_id}


# ═════════════════════════════════════════════════════════════════════════════
# SEARCH ENDPOINT
# ═════════════════════════════════════════════════════════════════════════════


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Semantic search",
    description="Search the knowledge base semantically and return ranked chunks with relevance scores.",
)
def semantic_search(payload: SearchRequest) -> SearchResponse:
    t_start = time.perf_counter()
    scored_chunks = search_with_scores(
        payload.query,
        top_k=payload.top_k,
        doc_ids=payload.doc_ids,
    )
    search_time = round((time.perf_counter() - t_start) * 1000, 1)

    results = [
        SearchResult(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            doc_name=chunk.doc_name,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page_number=chunk.page_number,
            relevance_score=round(score, 4),
            excerpt=chunk.text[:200],
        )
        for chunk, score in scored_chunks
    ]

    return SearchResponse(
        query=payload.query,
        results=results,
        total_results=len(results),
        search_time_ms=search_time,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CHAT ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Ask a question",
    description="Ask a question and get a grounded answer with source citations.",
)
@limiter.limit(settings.rate_limit_chat)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    session = get_or_create_session(payload.session_id)

    try:
        standalone_question = rewrite_query(session.messages, payload.question)
        result = answer_question(
            standalone_question,
            doc_id=payload.doc_id,
            doc_ids=payload.doc_ids,
            top_k=payload.top_k,
            answer_mode=payload.answer_mode,
        )
    except RAGBaseError:
        raise
    except Exception as e:
        logger.exception("Unexpected error answering question")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}") from e

    # Store original question (not rewritten) in history
    append_message(session.session_id, ChatMessage(role="user", content=payload.question))
    append_message(
        session.session_id,
        ChatMessage(role="assistant", content=result["answer"], sources=result["sources"]),
    )

    return ChatResponse(
        session_id=session.session_id,
        answer=result["answer"],
        sources=result["sources"],
        grounded=result["grounded"],
        warning=result["warning"],
        metrics=result.get("metrics"),
    )


@app.post(
    "/api/v1/chat/stream",
    tags=["Chat"],
    summary="Ask a question (streaming)",
    description="Same as /chat, but streams the answer token-by-token as newline-delimited JSON.",
)
@limiter.limit(settings.rate_limit_chat)
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """Fully async streaming endpoint. The event loop is never blocked."""
    session = get_or_create_session(payload.session_id)

    async def event_stream():
        full_answer = ""
        sources: list[dict] = []
        try:
            yield json.dumps({"type": "status", "text": "Analyzing question...", "session_id": session.session_id}) + "\n"

            # Rewrite query in background thread — non-blocking
            try:
                standalone_question = await rewrite_query_async(session.messages, payload.question)
            except Exception:
                standalone_question = payload.question

            # Stream tokens from async generator
            async for event in answer_question_stream_async(
                standalone_question,
                doc_id=payload.doc_id,
                doc_ids=payload.doc_ids,
                top_k=payload.top_k,
                answer_mode=payload.answer_mode,
            ):
                if event["type"] == "sources":
                    sources = event["sources"]
                elif event["type"] == "done":
                    full_answer = event["text"]
                yield json.dumps({**event, "session_id": session.session_id}) + "\n"

        except Exception as e:
            logger.exception("Streaming chat failed")
            yield json.dumps({
                "type": "error",
                "text": str(e),
                "session_id": session.session_id,
            }) + "\n"
            return

        # Save conversation history after stream completes
        append_message(session.session_id, ChatMessage(role="user", content=payload.question))
        append_message(
            session.session_id,
            ChatMessage(role="assistant", content=full_answer, sources=sources),
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ═════════════════════════════════════════════════════════════════════════════
# SESSION ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@app.get(
    "/api/v1/sessions",
    tags=["Sessions"],
    summary="List chat sessions",
    description="Returns all chat sessions, most recently active first.",
)
def get_sessions():
    return [s.model_dump() for s in list_sessions()]


@app.get(
    "/api/v1/sessions/{session_id}",
    tags=["Sessions"],
    summary="Get session details",
    description="Returns full session with all messages.",
)
def get_session_detail(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session found with id={session_id}")
    return session.model_dump()


@app.delete(
    "/api/v1/sessions/{session_id}",
    tags=["Sessions"],
    summary="Delete a session",
    description="Permanently delete a chat session and its history.",
)
def remove_session(session_id: str) -> dict:
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No session found with id={session_id}")
    return {"deleted": session_id}


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120, description="New session title")


@app.patch(
    "/api/v1/sessions/{session_id}/rename",
    tags=["Sessions"],
    summary="Rename a session",
    description="Update the title of a chat session.",
)
def rename_session_endpoint(session_id: str, payload: RenameRequest) -> dict:
    success = rename_session(session_id, payload.title)
    if not success:
        raise HTTPException(status_code=404, detail=f"No session found with id={session_id}")
    return {"session_id": session_id, "title": payload.title.strip()[:120]}


@app.get(
    "/api/v1/sessions/{session_id}/export",
    tags=["Sessions"],
    summary="Export a session",
    description="Export a chat session as Markdown or JSON.",
)
def export_session_endpoint(
    session_id: str,
    format: ExportFormat = Query(default=ExportFormat.MARKDOWN, description="Export format"),
):
    result = export_session(session_id, fmt=format)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No session found with id={session_id}")

    if format == ExportFormat.JSON:
        return JSONResponse(content=json.loads(result))
    else:
        return PlainTextResponse(
            content=result,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="session_{session_id}.md"'},
        )


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
def health() -> HealthResponse:
    return HealthResponse()


@app.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    tags=["System"],
    summary="System statistics",
    description="Returns document, chunk, session counts, and enhanced metrics.",
)
def stats() -> StatsResponse:
    kb = get_knowledge_base_summary()
    return StatsResponse(
        total_documents=kb["total_documents"],
        total_chunks=kb["total_chunks"],
        total_sessions=get_session_count(),
        total_words=kb["total_words"],
        total_file_size_bytes=kb["total_file_size_bytes"],
        avg_chunks_per_doc=kb["avg_chunks_per_doc"],
        storage_path=kb["storage_path"],
    )


@app.get(
    "/api/v1/suggestions",
    tags=["System"],
    summary="Smart prompt suggestions",
    description="Generate context-aware prompt suggestions based on uploaded documents.",
)
def get_suggestions():
    docs = list_documents()
    doc_names = [d["doc_name"] for d in docs]
    return generate_suggestions(doc_names)


# ─── Static Files (mounted LAST so API routes take priority) ──────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")