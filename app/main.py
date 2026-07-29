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

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.chat_memory import append_message, get_or_create_session
from app.ingestion import IngestionError, chunk_document, compute_file_hash
from app.models import ChatMessage, UploadResponse
from app.rag import RagError, answer_question, rewrite_query
from app.vectorstore import add_chunks, delete_document, find_document_by_hash, get_document, list_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
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
    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

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
        add_chunks(chunks)
    except IngestionError as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e)) from e
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
def chat(payload: ChatRequest) -> dict:
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
    }


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