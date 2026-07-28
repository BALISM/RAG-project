"""
Phase 2b — The vector store itself.

Chroma is doing two jobs for us:
  1. Storing each chunk's embedding vector alongside its original text and
     metadata (doc_id, doc_name, page_number).
  2. Similarity search: given a new vector (the embedded question), find
     the stored vectors "closest" to it. Closeness here is cosine
     similarity - basically, "do these two vectors point in a similar
     direction," which turns out to correlate well with "do these two
     pieces of text mean similar things."

Note this module never calls Gemini itself - it only stores/searches
vectors that embeddings.py already produced. Keeping "turn text into
numbers" and "store/search numbers" as separate modules means you could
swap Chroma for a different vector DB later without touching embeddings.py
at all, or swap the embedding model without touching this file.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.embeddings import embed_documents, embed_query
from app.models import DocumentChunk

_client = None
_COLLECTION_NAME = "documents"


def get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client.get_or_create_collection(_COLLECTION_NAME)


def add_chunks(chunks: list[DocumentChunk]) -> None:
    """Embed a batch of chunks and store them. This is the only place in
    the whole app that writes to the vector store."""
    if not chunks:
        return

    vectors = embed_documents([c.text for c in chunks])

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
            }
            for c in chunks
        ],
    )

    
def find_document_by_hash(content_hash: str) -> str | None:
    """Return the doc_id of an already-stored document with this exact
    content hash, or None if this file hasn't been uploaded before."""
    if not content_hash:
        return None
    results = get_collection().get(where={"content_hash": content_hash}, limit=1)
    ids = results.get("ids") or []
    if not ids:
        return None
    return results["metadatas"][0]["doc_id"]


def search(query: str, top_k: int | None = None, doc_id: str | None = None) -> list[DocumentChunk]:
    """Embed a question and return the top_k most similar stored chunks,
    ranked closest-first. Optionally restrict to a single doc_id."""
    k = top_k or settings.top_k_results
    query_vector = embed_query(query)

    where = {"doc_id": doc_id} if doc_id else None
    results = get_collection().query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where,
    )

    chunks = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    for chunk_id, text, meta in zip(ids, documents, metadatas):
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


def list_documents() -> list[dict]:
    """Distinct documents currently stored, with chunk counts - used by the
    frontend to show what's been uploaded so far."""
    collection = get_collection()
    all_items = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in all_items["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "doc_name": meta["doc_name"], "num_chunks": 0}
        seen[doc_id]["num_chunks"] += 1
    return list(seen.values())


def delete_document(doc_id: str) -> None:
    get_collection().delete(where={"doc_id": doc_id})