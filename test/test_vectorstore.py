"""
Formal test suite for vectorstore.py: storage, similarity search (with and
without doc_id/doc_ids filtering), duplicate detection, document detail
lookup, and delete behavior (including the on-disk file cleanup added in a
later commit).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import app.vectorstore as vectorstore
from app.models import DocumentChunk


def make_chunk(doc_id, chunk_index, text, doc_name="doc.txt", page_number=None,
                content_hash=None, stored_filename=None):
    return DocumentChunk(
        chunk_id=f"{doc_id}::chunk::{chunk_index}",
        doc_id=doc_id,
        doc_name=doc_name,
        chunk_index=chunk_index,
        text=text,
        page_number=page_number,
        content_hash=content_hash,
        stored_filename=stored_filename,
    )


# ---------------------------------------------------------------------------
# add_chunks + search - basic round trip
# ---------------------------------------------------------------------------

def test_add_and_search_round_trip(isolated_chroma, fake_embeddings):
    chunks = [make_chunk("d1", 0, "some content about cooking")]
    vectorstore.add_chunks(chunks)

    results = vectorstore.search("anything", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "d1::chunk::0"
    assert results[0].text == "some content about cooking"


def test_add_chunks_with_empty_list_does_nothing(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([])  # should not raise
    results = vectorstore.search("anything", top_k=5)
    assert results == []


def test_search_preserves_page_number(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([make_chunk("d1", 0, "text on page 3", page_number=3)])
    results = vectorstore.search("anything", top_k=5)
    assert results[0].page_number == 3


def test_search_with_no_page_number_returns_none_not_negative_one(isolated_chroma, fake_embeddings):
    # internal storage uses -1 as a sentinel for "no page" (Chroma metadata
    # can't store None) - this proves that sentinel never leaks back out.
    vectorstore.add_chunks([make_chunk("d1", 0, "no page here", page_number=None)])
    results = vectorstore.search("anything", top_k=5)
    assert results[0].page_number is None


# ---------------------------------------------------------------------------
# doc_id / doc_ids filtering
# ---------------------------------------------------------------------------

def test_search_filters_by_single_doc_id(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([
        make_chunk("docA", 0, "content A"),
        make_chunk("docB", 0, "content B"),
    ])
    results = vectorstore.search("anything", top_k=10, doc_id="docA")
    assert {r.doc_id for r in results} == {"docA"}


def test_search_filters_by_multiple_doc_ids(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([
        make_chunk("docA", 0, "content A"),
        make_chunk("docB", 0, "content B"),
        make_chunk("docC", 0, "content C"),
    ])
    results = vectorstore.search("anything", top_k=10, doc_ids=["docA", "docC"])
    found = {r.doc_id for r in results}
    assert found == {"docA", "docC"}
    assert "docB" not in found


def test_search_with_no_filter_searches_everything(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([
        make_chunk("docA", 0, "content A"),
        make_chunk("docB", 0, "content B"),
    ])
    results = vectorstore.search("anything", top_k=10)
    assert {r.doc_id for r in results} == {"docA", "docB"}


# ---------------------------------------------------------------------------
# find_document_by_hash - dedup
# ---------------------------------------------------------------------------

def test_find_document_by_hash_returns_none_when_not_found(isolated_chroma, fake_embeddings):
    assert vectorstore.find_document_by_hash("nonexistent-hash") is None


def test_find_document_by_hash_returns_none_for_empty_string(isolated_chroma, fake_embeddings):
    assert vectorstore.find_document_by_hash("") is None


def test_find_document_by_hash_finds_matching_document(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([make_chunk("d1", 0, "content", content_hash="abc123")])
    assert vectorstore.find_document_by_hash("abc123") == "d1"


def test_find_document_by_hash_ignores_different_hash(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([make_chunk("d1", 0, "content", content_hash="abc123")])
    assert vectorstore.find_document_by_hash("xyz789") is None


# ---------------------------------------------------------------------------
# get_document / list_documents
# ---------------------------------------------------------------------------

def test_get_document_returns_none_for_unknown_id(isolated_chroma, fake_embeddings):
    assert vectorstore.get_document("nonexistent") is None


def test_get_document_returns_all_chunks_in_order(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([
        make_chunk("d1", 2, "third chunk"),
        make_chunk("d1", 0, "first chunk"),
        make_chunk("d1", 1, "second chunk"),
    ])
    doc = vectorstore.get_document("d1")
    assert doc["num_chunks"] == 3
    ordered_texts = [c["text"] for c in doc["chunks"]]
    assert ordered_texts == ["first chunk", "second chunk", "third chunk"]


def test_list_documents_groups_by_doc_id_with_correct_counts(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([
        make_chunk("d1", 0, "a", doc_name="one.txt"),
        make_chunk("d1", 1, "b", doc_name="one.txt"),
        make_chunk("d2", 0, "c", doc_name="two.txt"),
    ])
    docs = {d["doc_id"]: d for d in vectorstore.list_documents()}
    assert docs["d1"]["num_chunks"] == 2
    assert docs["d1"]["doc_name"] == "one.txt"
    assert docs["d2"]["num_chunks"] == 1


def test_list_documents_empty_store_returns_empty_list(isolated_chroma, fake_embeddings):
    assert vectorstore.list_documents() == []


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------

def test_delete_document_removes_only_that_document(isolated_chroma, fake_embeddings):
    vectorstore.add_chunks([
        make_chunk("docA", 0, "keep this"),
        make_chunk("docB", 0, "delete this"),
    ])
    vectorstore.delete_document("docB")
    remaining = vectorstore.list_documents()
    assert len(remaining) == 1
    assert remaining[0]["doc_id"] == "docA"


def test_delete_document_removes_the_file_from_disk(isolated_chroma, fake_embeddings, tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))

    real_file = settings.upload_path / "test_upload.txt"
    real_file.write_text("some content")
    assert real_file.exists()

    vectorstore.add_chunks([make_chunk("d1", 0, "content", stored_filename="test_upload.txt")])
    vectorstore.delete_document("d1")

    assert not real_file.exists(), "delete_document should remove the actual file, not just the vectors"


def test_delete_document_on_unknown_id_does_not_raise(isolated_chroma, fake_embeddings):
    vectorstore.delete_document("this-was-never-added")  # should be a silent no-op


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
    #new file