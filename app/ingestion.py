"""
Document ingestion — text extraction and overlapping chunking.

Supports PDF, DOCX, TXT, and MD files.  Extracts text with page-level
tracking (PDFs), normalizes whitespace, computes content hashes for
deduplication, and splits text into overlapping word-count windows for
optimal retrieval precision.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path

import pypdf
from docx import Document as DocxDocument

from app.config import settings
from app.exceptions import IngestionError
from app.logging_config import get_logger, log_duration
from app.models import DocumentChunk

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def compute_file_hash(path: Path) -> str:
    """SHA-256 of the raw file bytes, used for duplicate detection."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()


# ─── Text Normalization ──────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """Collapse excessive whitespace and strip control characters, keeping
    readability intact.  This prevents garbage characters from polluting
    embeddings and ensures consistent chunking behavior."""
    # Remove control chars except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse runs of 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of whitespace (excluding newlines) to single space
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


# ─── Text Extraction ─────────────────────────────────────────────────────────


def _extract_pdf_pages(path: Path) -> list[tuple[int | None, str]]:
    reader = pypdf.PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = _normalize_text(text)
        if text:
            pages.append((i + 1, text))  # 1-indexed page numbers
    return pages


def _extract_docx_text(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_pages(path: Path) -> list[tuple[int | None, str]]:
    """Returns a list of (page_number_or_None, text) tuples.

    PDFs get one entry per page; DOCX/TXT/MD get a single entry with
    page_number=None.
    """
    ext = path.suffix.lower()

    if ext == ".pdf":
        pages = _extract_pdf_pages(path)
        if not pages:
            raise IngestionError("No extractable text found in PDF (it may be scanned/image-only)")
        return pages

    if ext == ".docx":
        text = _normalize_text(_extract_docx_text(path))
        if not text:
            raise IngestionError("No extractable text found in DOCX")
        return [(None, text)]

    if ext in (".txt", ".md"):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = _normalize_text(raw)
        if not text:
            raise IngestionError("File is empty")
        return [(None, text)]

    raise IngestionError(
        f"Unsupported file type: {ext}",
        detail=f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}",
    )


# ─── Chunking ────────────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size_words: int | None = None,
    overlap_words: int | None = None,
) -> list[str]:
    """Split text into overlapping word-count windows."""
    size = chunk_size_words or settings.chunk_size_words
    overlap = overlap_words if overlap_words is not None else settings.chunk_overlap_words

    if overlap >= size:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = size - overlap
    start = 0
    while start < len(words):
        window = words[start : start + size]
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
        start += step

    return chunks


# ─── Full Pipeline ───────────────────────────────────────────────────────────


def chunk_document(
    path: Path,
    doc_name: str | None = None,
) -> tuple[str, list[DocumentChunk]]:
    """Full ingestion pipeline for one file: extract → normalize → chunk →
    attach metadata.  Returns (doc_id, list_of_chunks)."""
    doc_id = uuid.uuid4().hex[:12]
    doc_name = doc_name or path.name
    file_hash = compute_file_hash(path)

    with log_duration(logger, f"Ingesting '{doc_name}'"):
        pages = extract_pages(path)

        all_chunks: list[DocumentChunk] = []
        chunk_index = 0
        for page_number, page_text in pages:
            for piece in chunk_text(page_text):
                piece_clean = piece.strip()
                if not piece_clean:
                    continue
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}::chunk::{chunk_index}",
                        doc_id=doc_id,
                        doc_name=doc_name,
                        chunk_index=chunk_index,
                        text=piece_clean,
                        page_number=page_number,
                        content_hash=file_hash,
                        stored_filename=path.name,
                    )
                )
                chunk_index += 1

    if not all_chunks:
        raise IngestionError("Document produced zero chunks after extraction")

    logger.info(
        "Ingested '%s' → doc_id=%s, %d chunks, hash=%s…",
        doc_name, doc_id, len(all_chunks), file_hash[:12],
    )
    return doc_id, all_chunks


def get_file_metadata(path: Path) -> dict:
    """Extract metadata about a file without fully processing it."""
    stat = path.stat()
    return {
        "file_size_bytes": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
    }