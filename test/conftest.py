"""
Shared pytest fixtures.

isolated_chroma: fresh temp vector store per test, no shared state.
fake_embeddings: deterministic stand-in vectors, no API key needed.
isolated_sessions_db: fresh SQLite per test for chat memory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import app.chat_memory as memory
import app.vectorstore as vectorstore
from app.config import settings


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    """Point the vector store at a fresh tmp directory for isolation."""
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(vectorstore, "_client", None)
    yield
    monkeypatch.setattr(vectorstore, "_client", None)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Replace real Gemini embedding calls with cheap deterministic fakes."""
    def fake_embed_documents(texts):
        return [[float(len(t) % 10), 1.0, 0.0] for t in texts]

    def fake_embed_query(text):
        return [float(len(text) % 10), 1.0, 0.0]

    monkeypatch.setattr(vectorstore, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(vectorstore, "embed_query", fake_embed_query)


@pytest.fixture
def isolated_sessions_db(tmp_path, monkeypatch):
    """Fresh SQLite database for chat memory tests."""
    monkeypatch.setattr(settings, "sessions_db", str(tmp_path / "test_sessions.db"))
    monkeypatch.setattr(memory, "_connection", None)
    yield
    monkeypatch.setattr(memory, "_connection", None)