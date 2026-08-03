"""
Test suite for rag.py — grounding checks, context formatting, source
building, and query rewriting logic (everything that doesn't need a live
Gemini API call).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.models import ChatMessage, DocumentChunk
from app.rag import _build_sources, _format_context, check_grounding


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_chunk(doc_id="d1", index=0, text="sample text", doc_name="doc.pdf", page=None):
    return DocumentChunk(
        chunk_id=f"{doc_id}::chunk::{index}",
        doc_id=doc_id,
        doc_name=doc_name,
        chunk_index=index,
        text=text,
        page_number=page,
    )


# ─── check_grounding ─────────────────────────────────────────────────────────

class TestCheckGrounding:
    def test_no_sources_always_grounded(self):
        result = check_grounding("Anything here.", num_sources=0)
        assert result["grounded"] is True
        assert result["warning"] is None

    def test_answer_with_citation_is_grounded(self):
        result = check_grounding("As stated in [Source 1], the answer is 42.", num_sources=3)
        assert result["grounded"] is True

    def test_answer_with_refusal_is_grounded(self):
        result = check_grounding("The documents don't contain information about that.", num_sources=3)
        assert result["grounded"] is True

    def test_answer_without_citation_or_refusal_is_not_grounded(self):
        result = check_grounding("The answer is 42.", num_sources=3)
        assert result["grounded"] is False
        assert result["warning"] is not None

    def test_case_insensitive_citation_detection(self):
        result = check_grounding("According to [source 2], yes.", num_sources=2)
        assert result["grounded"] is True

    def test_refusal_phrase_not_mentioned(self):
        result = check_grounding("not mentioned in the uploaded files.", num_sources=1)
        assert result["grounded"] is True

    def test_refusal_cannot_find(self):
        result = check_grounding("I cannot find that information in the documents.", num_sources=2)
        assert result["grounded"] is True

    def test_empty_answer_is_not_grounded(self):
        result = check_grounding("", num_sources=2)
        assert result["grounded"] is False


# ─── _format_context ─────────────────────────────────────────────────────────

class TestFormatContext:
    def test_single_chunk_without_page(self):
        chunks = [_make_chunk(text="Hello world", doc_name="a.txt")]
        ctx = _format_context(chunks)
        assert "[Source 1]" in ctx
        assert "(from a.txt)" in ctx
        assert "Hello world" in ctx

    def test_single_chunk_with_page(self):
        chunks = [_make_chunk(text="Page content", doc_name="b.pdf", page=5)]
        ctx = _format_context(chunks)
        assert "[Source 1]" in ctx
        assert "page 5" in ctx

    def test_multiple_chunks_numbered_sequentially(self):
        chunks = [
            _make_chunk(index=0, text="First", doc_name="a.txt"),
            _make_chunk(index=1, text="Second", doc_name="a.txt"),
            _make_chunk(index=2, text="Third", doc_name="b.pdf", page=3),
        ]
        ctx = _format_context(chunks)
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx
        assert "[Source 3]" in ctx

    def test_empty_chunks_returns_empty_string(self):
        assert _format_context([]) == ""


# ─── _build_sources ──────────────────────────────────────────────────────────

class TestBuildSources:
    def test_returns_correct_fields(self):
        chunks = [_make_chunk(text="A" * 300, doc_name="test.pdf", page=2)]
        sources = _build_sources(chunks)
        assert len(sources) == 1
        s = sources[0]
        assert s["doc_name"] == "test.pdf"
        assert s["page_number"] == 2
        assert s["source_index"] == 1
        assert len(s["excerpt"]) <= 200

    def test_excerpt_truncates_long_text(self):
        long_text = "word " * 100
        chunks = [_make_chunk(text=long_text)]
        sources = _build_sources(chunks)
        assert len(sources[0]["excerpt"]) == 200

    def test_multiple_chunks_have_sequential_indices(self):
        chunks = [_make_chunk(index=i, text=f"text {i}") for i in range(4)]
        sources = _build_sources(chunks)
        assert [s["source_index"] for s in sources] == [1, 2, 3, 4]

    def test_none_page_number_preserved(self):
        chunks = [_make_chunk(page=None)]
        sources = _build_sources(chunks)
        assert sources[0]["page_number"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
