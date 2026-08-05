"""
FastAPI application — the HTTP layer connecting everything together.

Architecture:
  - All API routes prefixed with /api/v1/ for versioning
  - Static files served at / for the browser UI
  - Structured error handling via custom exception classes
  - Security headers middleware
  - Rate limiting on costly endpoints (upload, chat)
  - Rich OpenAPI metadata for auto-generated docs
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.chat_memory import (
    append_message,
    delete_session,
    get_or_create_session,
    get_session,
    get_session_count,
    list_sessions,
)
from app.config import settings
from app.exceptions import IngestionError, RAGBaseError
from app.ingestion import SUPPORTED_EXTENSIONS, chunk_document, compute_file_hash, get_file_metadata
from app.logging_config import get_logger, setup_logging
from app.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    StatsResponse,
    UploadResponse,
)
from app.rag import GenerationError, answer_question, answer_question_stream, rewrite_query
from app.vectorstore import (
    add_chunks,
    delete_document,
    find_document_by_hash,
    get_collection_stats,
    get_document,
    list_documents,
)

# ─── Logging Setup ────────────────────────────────────────────────────────────

setup_logging(settings.log_level)
logger = get_logger(__name__)

# ─── Rate Limiter ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ─── App Creation ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "Upload documents, then ask questions grounded in exactly what's in them. "
        "Built with FastAPI, Google Gemini, and ChromaDB."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Documents", "description": "Upload, list, inspect, and delete documents"},
        {"name": "Chat", "description": "Ask questions with RAG-grounded answers"},
        {"name": "Sessions", "description": "Manage conversation sessions"},
        {"name": "System", "description": "Health checks and system statistics"},
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
    logger.info("🚀 RAG Chatbot API v1.0.0 started")
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
    )


@app.post(
    "/api/v1/chat/stream",
    tags=["Chat"],
    summary="Ask a question (streaming)",
    description="Same as /chat, but streams the answer token-by-token as newline-delimited JSON.",
)
@limiter.limit(settings.rate_limit_chat)
def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    session = get_or_create_session(payload.session_id)

    def event_stream():
        full_answer = ""
        sources: list[dict] = []
        try:
            yield json.dumps({"type": "status", "text": "Analyzing conversation history..."}) + "\n"
            try:
                standalone_question = rewrite_query(session.messages, payload.question)
            except Exception:
                standalone_question = payload.question

            for event in answer_question_stream(
                standalone_question,
                doc_id=payload.doc_id,
                doc_ids=payload.doc_ids,
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

        # Save after stream completes successfully
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
    description="Returns document, chunk, and session counts.",
)
def stats() -> StatsResponse:
    vs_stats = get_collection_stats()
    return StatsResponse(
        total_documents=vs_stats["total_documents"],
        total_chunks=vs_stats["total_chunks"],
        total_sessions=get_session_count(),
        storage_path=vs_stats["storage_path"],
    )


# ─── Static Files (mounted LAST so API routes take priority) ──────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")