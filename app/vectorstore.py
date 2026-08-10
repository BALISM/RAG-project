"""
Vector store layer — ChromaDB wrapper for storage, search, and lifecycle.

This module never calls the embedding API directly for generation — it only
stores/searches vectors that embeddings.py produced.  This separation means
you can swap ChromaDB for Pinecone/Weaviate/pgvector without touching the
embedding layer, or swap the embedding model without touching this file.

Enhanced:
  - Relevance threshold filtering in search results
  - Document preview (first N chunks)
  - Richer knowledge base statistics
"""
from __future__ import annotations

from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.embeddings import embed_documents, embed_query
from app.exceptions import VectorStoreError
from app.logging_config import get_logger, log_duration
from app.models import DocumentChunk

logger = get_logger(__name__)

_client = None
_COLLECTION_NAME = "documents"


def get_collection():
    """Get or create the ChromaDB collection singleton."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client.get_or_create_collection(_COLLECTION_NAME)


# ─── Write Operations ────────────────────────────────────────────────────────


def add_chunks(
    chunks: list[DocumentChunk],
    file_size_bytes: int | None = None,
) -> None:
    """Embed a batch of chunks and store them.  This is the only place in
    the whole app that writes to the vector store."""
    if not chunks:
        return

    with log_duration(logger, f"Storing {len(chunks)} chunks for '{chunks[0].doc_name}'"):
        vectors = embed_documents([c.text for c in chunks])

        # Calculate total word count across all chunks
        total_words = sum(len(c.text.split()) for c in chunks)
        upload_time = datetime.now().isoformat()

        get_collection().add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number if c.page_number is not None else -1,
                    "content_hash": c.content_hash or "",
                    "stored_filename": c.stored_filename or "",
                    "uploaded_at": upload_time,
                    "file_size_bytes": file_size_bytes or 0,
                    "word_count": total_words,
                }
                for c in chunks
            ],
        )


# ─── Search Operations ───────────────────────────────────────────────────────


def search(
    query: str,
    top_k: int | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[DocumentChunk]:
    """Embed a question and return the top_k most similar stored chunks,
    ranked closest-first.  Optionally restrict to specific documents."""
    k = top_k or settings.top_k_results
    query_vector = embed_query(query)

    where = None
    if doc_ids:
        where = {"doc_id": {"$in": doc_ids}}
    elif doc_id:
        where = {"doc_id": doc_id}

    results = get_collection().query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    for i, (chunk_id, text, meta) in enumerate(zip(ids, documents, metadatas)):
        page_number = meta["page_number"]
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=meta["doc_id"],
                doc_name=meta["doc_name"],
                chunk_index=meta["chunk_index"],
                text=text,
                page_number=None if page_number == -1 else page_number,
            )
        )
    return chunks


def search_with_scores(
    query: str,
    top_k: int | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    apply_threshold: bool = False,
) -> list[tuple[DocumentChunk, float]]:
    """Like search(), but also returns the relevance score for each chunk."""
    k = top_k or settings.top_k_results
    query_vector = embed_query(query)

    where = None
    if doc_ids:
        where = {"doc_id": {"$in": doc_ids}}
    elif doc_id:
        where = {"doc_id": doc_id}

    results = get_collection().query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    pairs = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    for i, (chunk_id, text, meta) in enumerate(zip(ids, documents, metadatas)):
        page_number = meta["page_number"]
        distance = distances[i] if i < len(distances) else 0.0
        similarity = 1.0 / (1.0 + distance)

        # Apply relevance threshold filtering
        if apply_threshold and similarity < settings.relevance_threshold:
            continue

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            doc_name=meta["doc_name"],
            chunk_index=meta["chunk_index"],
            text=text,
            page_number=None if page_number == -1 else page_number,
        )
        pairs.append((chunk, similarity))
    return pairs


# ─── Dedup ────────────────────────────────────────────────────────────────────


def find_document_by_hash(content_hash: str) -> str | None:
    """Return the doc_id of an already-stored document with this exact
    content hash, or None if not found."""
    if not content_hash:
        return None
    results = get_collection().get(where={"content_hash": content_hash}, limit=1)
    ids = results.get("ids") or []
    if not ids:
        return None
    return results["metadatas"][0]["doc_id"]


# ─── Document Detail & Listing ────────────────────────────────────────────────


def get_document(doc_id: str) -> dict | None:
    """Full detail for one document: metadata plus every chunk, in order."""
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    ids = results.get("ids") or []
    if not ids:
        return None

    chunks = []
    doc_name = ""
    uploaded_at = None
    file_size_bytes = 0
    word_count = 0

    for chunk_id, text, meta in zip(ids, results["documents"], results["metadatas"]):
        doc_name = meta["doc_name"]
        uploaded_at = meta.get("uploaded_at")
        file_size_bytes = meta.get("file_size_bytes", 0)
        word_count = meta.get("word_count", 0)
        page_number = meta["page_number"]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": meta["chunk_index"],
                "page_number": None if page_number == -1 else page_number,
                "text": text,
            }
        )
    chunks.sort(key=lambda c: c["chunk_index"])

    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "uploaded_at": uploaded_at,
        "file_size_bytes": file_size_bytes,
        "word_count": word_count,
    }


def list_documents() -> list[dict]:
    """Distinct documents currently stored, with chunk counts and metadata."""
    collection = get_collection()
    all_items = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in all_items["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "doc_name": meta["doc_name"],
                "num_chunks": 0,
                "uploaded_at": meta.get("uploaded_at"),
                "file_size_bytes": meta.get("file_size_bytes", 0),
                "word_count": meta.get("word_count", 0),
            }
        seen[doc_id]["num_chunks"] += 1
    return list(seen.values())


# ─── Delete ───────────────────────────────────────────────────────────────────


def delete_document(doc_id: str) -> None:
    """Remove both the stored vectors AND the original uploaded file on disk."""
    collection = get_collection()
    existing = collection.get(where={"doc_id": doc_id}, limit=1, include=["metadatas"])
    metadatas = existing.get("metadatas") or []

    collection.delete(where={"doc_id": doc_id})

    if metadatas:
        stored_filename = metadatas[0].get("stored_filename")
        if stored_filename:
            file_path = settings.upload_path / stored_filename
            file_path.unlink(missing_ok=True)
            logger.info("Deleted document %s and file %s", doc_id, stored_filename)
        else:
            logger.info("Deleted document %s (no file on disk)", doc_id)


# ─── Statistics ───────────────────────────────────────────────────────────────


def get_collection_stats() -> dict:
    """Return statistics about the vector store collection."""
    collection = get_collection()
    count = collection.count()
    docs = list_documents()
    return {
        "total_chunks": count,
        "total_documents": len(docs),
        "storage_path": str(settings.chroma_path),
    }


# ─── New: Document Preview ───────────────────────────────────────────────────


def get_document_preview(doc_id: str, max_chunks: int = 3) -> dict | None:
    """Return the first N chunks of a document for inline preview.
    Lighter than get_document() since it doesn't return all chunks."""
    full = get_document(doc_id)
    if full is None:
        return None

    preview_chunks = full["chunks"][:max_chunks]
    return {
        "doc_id": full["doc_id"],
        "doc_name": full["doc_name"],
        "num_chunks": full["num_chunks"],
        "preview_chunks": preview_chunks,
        "uploaded_at": full.get("uploaded_at"),
        "file_size_bytes": full.get("file_size_bytes", 0),
        "word_count": full.get("word_count", 0),
    }


# ─── New: Enhanced Knowledge Base Summary ────────────────────────────────────


def get_knowledge_base_summary() -> dict:
    """Enhanced statistics including word counts, file sizes, and averages."""
    docs = list_documents()
    total_chunks = sum(d["num_chunks"] for d in docs)
    total_words = sum(d.get("word_count", 0) for d in docs)
    total_file_size = sum(d.get("file_size_bytes", 0) for d in docs)
    avg_chunks = total_chunks / len(docs) if docs else 0.0

    return {
        "total_documents": len(docs),
        "total_chunks": total_chunks,
        "total_words": total_words,
        "total_file_size_bytes": total_file_size,
        "avg_chunks_per_doc": round(avg_chunks, 1),
        "storage_path": str(settings.chroma_path),
        "documents": docs,
    }