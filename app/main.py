"""
Phase 3 — FastAPI routes.

This phase doesn't introduce any new RAG concepts - it's pure plumbing,
connecting Phase 1 (ingestion.py) and Phase 2 (embeddings.py/vectorstore.py)
to the outside world via HTTP. The interesting part is already built; this
phase just makes it reachable.

Endpoints:
  POST   /documents/upload   Upload a file -> extract -> chunk -> embed -> store
  GET    /documents          List what's currently in the vector store
  DELETE /documents/{doc_id} Remove a document and all its chunks
  GET    /health
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.chat_memory import append_message, get_or_create_session
from app.ingestion import IngestionError, chunk_document, compute_file_hash
from app.models import ChatMessage, UploadResponse
from app.rag import RagError, answer_question, answer_question_stream, rewrite_query
from app.vectorstore import add_chunks, delete_document, find_document_by_hash, get_document, list_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting is per-IP, keyed off the request's remote address. Only
# applied to endpoints that cost real money/quota (uploads embed every
# chunk; chat calls Gemini at least once, twice if query rewriting kicks
# in) - read-only endpoints like GET /documents stay unrestricted since
# they're just local Chroma lookups with no external cost.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="RAG Chatbot", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@app.post("/documents/upload", response_model=UploadResponse)
@limiter.limit(settings.rate_limit_upload)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Save the upload to disk under a random name so two people uploading
    # "notes.txt" at the same time can't collide with each other.
    temp_name = f"{uuid.uuid4().hex[:12]}{suffix}"
    dest_path = settings.upload_path / temp_name
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    try:
        bytes_written = 0
        with dest_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # read 1MB at a time
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
        existing_doc_id = find_document_by_hash(compute_file_hash(dest_path))
        if existing_doc_id:
            dest_path.unlink(missing_ok=True)
            logger.info("Duplicate upload of %s detected -> reusing doc_id=%s", file.filename, existing_doc_id)
            existing = next((d for d in list_documents() if d["doc_id"] == existing_doc_id), None)
            return UploadResponse(
                doc_id=existing_doc_id,
                doc_name=existing["doc_name"] if existing else (file.filename or temp_name),
                num_chunks=existing["num_chunks"] if existing else 0,
                num_pages=None,
            )

        doc_id, chunks = chunk_document(dest_path, doc_name=file.filename or temp_name)

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

        add_chunks(chunks)
    except IngestionError as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        # Re-raise as-is (e.g. the 403 storage-limit check above) - without
        # this, the generic `except Exception` below would catch it too
        # (HTTPException IS an Exception) and wrap a clean 403 into a
        # confusing 500. Cleanup already happened at each raise site.
        raise
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        logger.exception("Unexpected error ingesting %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}") from e

    page_numbers = {c.page_number for c in chunks if c.page_number is not None}
    logger.info("Ingested %s -> doc_id=%s, %d chunks", file.filename, doc_id, len(chunks))

    return UploadResponse(
        doc_id=doc_id,
        doc_name=file.filename or temp_name,
        num_chunks=len(chunks),
        num_pages=max(page_numbers) if page_numbers else None,
    )


@app.get("/documents")
def get_documents() -> list[dict]:
    return list_documents()


@app.get("/documents/{doc_id}")
def get_document_detail(doc_id: str) -> dict:
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No document found with doc_id={doc_id}")
    return doc


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None      # omit on the first message; reuse the returned one after that
    doc_id: str | None = None          # optional: restrict search to one document
    doc_ids: list[str] | None = None   # optional: restrict search to a specific set of documents


@app.post("/chat")
@limiter.limit(settings.rate_limit_chat)
def chat(request: Request, payload: ChatRequest) -> dict:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    session = get_or_create_session(payload.session_id)

    try:
        # Rewrite using history BEFORE retrieval - this is the whole point:
        # search needs a standalone question, not "what about his education?"
        standalone_question = rewrite_query(session.messages, payload.question)

        result = answer_question(standalone_question, doc_id=payload.doc_id, doc_ids=payload.doc_ids)
    except RagError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error answering question")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}") from e

    # Store what the user actually typed (not the rewritten version) so the
    # history reads naturally if it's ever displayed - the rewrite is an
    # internal retrieval detail, not something the user said.
    append_message(session.session_id, ChatMessage(role="user", content=payload.question))
    append_message(
        session.session_id,
        ChatMessage(role="assistant", content=result["answer"], sources=result["sources"]),
    )

    return {
        "session_id": session.session_id,
        "answer": result["answer"],
        "sources": result["sources"],
        "grounded": result["grounded"],
        "warning": result["warning"],
    }


@app.post("/chat/stream")
@limiter.limit(settings.rate_limit_chat)
def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """Same logic as /chat, but streams the answer as newline-delimited
    JSON events instead of waiting for the whole thing. One JSON object per
    line: {"type": "sources"|"token"|"done"|"error", ...}. Newline-delimited
    JSON rather than real SSE (text/event-stream) because it's simpler to
    both produce here and consume in the frontend with plain fetch() -
    no EventSource, no "data: " prefix parsing, just split on newlines."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    session = get_or_create_session(payload.session_id)
    standalone_question = rewrite_query(session.messages, payload.question)

    def event_stream():
        full_answer = ""
        sources: list[dict] = []
        try:
            for event in answer_question_stream(
                standalone_question, doc_id=payload.doc_id, doc_ids=payload.doc_ids
            ):
                if event["type"] == "sources":
                    sources = event["sources"]
                elif event["type"] == "done":
                    full_answer = event["text"]
                yield json.dumps({**event, "session_id": session.session_id}) + "\n"
        except Exception as e:
            logger.exception("Streaming chat failed")
            yield json.dumps({"type": "error", "text": str(e), "session_id": session.session_id}) + "\n"
            return

        # Save to memory only after the stream finishes successfully - an
        # interrupted/failed stream shouldn't leave a half-answer in history.
        append_message(session.session_id, ChatMessage(role="user", content=payload.question))
        append_message(
            session.session_id,
            ChatMessage(role="assistant", content=full_answer, sources=sources),
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str) -> dict:
    delete_document(doc_id)
    return {"deleted": doc_id}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Mounted LAST and at "/" on purpose: Starlette matches routes in the order
# they were registered, so every API route above still wins over this catch-
# all. If this were mounted first, it would swallow requests to /chat,
# /documents, etc. before they ever reached those handlers.
app.mount("/", StaticFiles(directory="static", html=True), name="static")