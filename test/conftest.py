"""
Shared pytest fixtures.

isolated_chroma: points the vector store at a fresh tmp_path for the
duration of one test, and resets the cached client so a stale connection
from a previous test can never leak in. Without this, tests would all
share the same on-disk chroma_db/ and step on each other's data.

fake_embeddings: replaces the real Gemini embedding calls with a cheap
deterministic stand-in, so the test suite runs instantly and needs no API
key. Tests that care about specific vector *values* pass their own vectors
in directly; this fixture is for tests that only care about storage/
retrieval plumbing working correctly, not about real semantic similarity.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import app.vectorstore as vectorstore
from app.config import settings


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(vectorstore, "_client", None)
    yield
    monkeypatch.setattr(vectorstore, "_client", None)


@pytest.fixture
def fake_embeddings(monkeypatch):
    def fake_embed_documents(texts):
        return [[float(len(t) % 10), 1.0, 0.0] for t in texts]

    def fake_embed_query(text):
        return [float(len(text) % 10), 1.0, 0.0]

    monkeypatch.setattr(vectorstore, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(vectorstore, "embed_query", fake_embed_query)
    