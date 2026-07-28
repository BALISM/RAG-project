"""
Phase 1 — Document ingestion.

Two jobs:
  1. extract_pages(): pull text out of a PDF/DOCX/TXT file, keeping track of
     which page each bit of text came from (PDFs only - DOCX/TXT have no
     page concept, so page_number stays None for those).
  2. chunk_text(): split text into overlapping word-count windows.

Why overlap? If a sentence explaining something important happens to fall
right on a chunk boundary, a non-overlapping chunker can split it so that
NEITHER resulting chunk contains the full idea, and a search for that idea
might not match either chunk well. Overlap means the boundary region gets
duplicated across two chunks, so at least one of them holds the complete
thought. The trade-off is storing (and later, embedding) a bit more text
than the source actually contains - that's the whole trade-off, and it's
usually worth it.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pypdf
from docx import Document as DocxDocument

from app.config import settings
from app.models import DocumentChunk
import hashlib
class IngestionError(Exception):
    pass


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_pdf_pages(path: Path) -> list[tuple[int | None, str]]:
    reader = pypdf.PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))  # 1-indexed page numbers for humans
    return pages


def _extract_docx_text(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_pages(path: Path) -> list[tuple[int | None, str]]:
    """Returns a list of (page_number_or_None, text) tuples. PDFs get one
    entry per page; DOCX/TXT get a single entry with page_number=None."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = _extract_pdf_pages(path)
        if not pages:
            raise IngestionError("No extractable text found in PDF (it may be scanned/image-only)")
        return pages
    if ext == ".docx":
        text = _extract_docx_text(path)
        if not text.strip():
            raise IngestionError("No extractable text found in DOCX")
        return [(None, text)]
    if ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            raise IngestionError("File is empty")
        return [(None, text)]
    raise IngestionError(f"Unsupported file type: {ext} (supported: .pdf, .docx, .txt, .md)")


# ---------------------------------------------------------------------------
# Chunking — sliding window over words, with overlap
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size_words: int | None = None,
    overlap_words: int | None = None,
) -> list[str]:
    """Split text into overlapping word-count windows. Returns a list of
    chunk strings (no metadata yet - that gets attached in chunk_document)."""
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


def chunk_document(
    path: Path,
    doc_name: str | None = None,
) -> tuple[str, list[DocumentChunk]]:
    """Full Phase 1 pipeline for one file: extract -> chunk -> attach
    metadata. Returns (doc_id, list_of_chunks)."""
    doc_id = uuid.uuid4().hex[:12]
    doc_name = doc_name or path.name
    pages = extract_pages(path)

    all_chunks: list[DocumentChunk] = []
    chunk_index = 0
    for page_number, page_text in pages:
        for piece in chunk_text(page_text):
            all_chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}::chunk::{chunk_index}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunk_index=chunk_index,
                    text=piece,
                    page_number=page_number,
                )
            )
            chunk_index += 1

    if not all_chunks:
        raise IngestionError("Document produced zero chunks after extraction")

    return doc_id, all_chunks