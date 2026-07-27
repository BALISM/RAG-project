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

from app.config import settings
from app.ingestion import IngestionError, chunk_document
from app.models import UploadResponse
from app.vectorstore import add_chunks, delete_document, list_documents

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


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str) -> dict:
    delete_document(doc_id)
    return {"deleted": doc_id}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}