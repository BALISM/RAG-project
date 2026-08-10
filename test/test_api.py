"""
API integration tests for FastAPI routes (v1 endpoints).
Uses FastAPI TestClient to test all document, chat, session, and system endpoints.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
import app.vectorstore as vectorstore
import app.chat_memory as memory
from app.config import settings

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_isolated_env(tmp_path, monkeypatch):
    """Ensure every API test runs with clean isolated DBs and fake embeddings."""
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "sessions_db", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(vectorstore, "_client", None)
    monkeypatch.setattr(memory, "_connection", None)

    def fake_embed_documents(texts):
        return [[float(len(t) % 10), 1.0, 0.0] for t in texts]

    def fake_embed_query(text):
        return [float(len(text) % 10), 1.0, 0.0]

    monkeypatch.setattr(vectorstore, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(vectorstore, "embed_query", fake_embed_query)

    yield

    monkeypatch.setattr(vectorstore, "_client", None)
    monkeypatch.setattr(memory, "_connection", None)


# ─── System Endpoints ─────────────────────────────────────────────────────────

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"


def test_stats_endpoint():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "total_sessions" in data


# ─── Document Endpoints ───────────────────────────────────────────────────────

def test_get_documents_empty():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_upload_document_success(tmp_path):
    file_content = b"This is a test document with enough words to produce chunks cleanly."
    files = {"file": ("test.txt", file_content, "text/plain")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["doc_name"] == "test.txt"
    assert data["num_chunks"] > 0
    assert "doc_id" in data


def test_upload_document_unsupported_extension():
    files = {"file": ("test.xyz", b"some data", "application/octet-stream")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_get_document_detail_not_found():
    response = client.get("/api/v1/documents/nonexistent")
    assert response.status_code == 404


# ─── Session Endpoints ────────────────────────────────────────────────────────

def test_list_and_delete_sessions():
    # List sessions initially empty
    resp = client.get("/api/v1/sessions")
    assert resp.status_code == 200
    assert resp.json() == []

    # Create a session directly via memory
    sess = memory.create_session()
    resp = client.get("/api/v1/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["session_id"] == sess.session_id

    # Delete session
    del_resp = client.delete(f"/api/v1/sessions/{sess.session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] == sess.session_id


def test_rename_and_export_session():
    sess = memory.create_session()

    # Rename session
    rename_resp = client.patch(
        f"/api/v1/sessions/{sess.session_id}/rename",
        json={"title": "Renamed Session Title"},
    )
    assert rename_resp.status_code == 200
    assert rename_resp.json()["title"] == "Renamed Session Title"

    # Export session
    export_resp = client.get(f"/api/v1/sessions/{sess.session_id}/export?format=markdown")
    assert export_resp.status_code == 200
    assert "Renamed Session Title" in export_resp.text


# ─── Search & Preview Endpoints ──────────────────────────────────────────────

def test_semantic_search_endpoint():
    resp = client.post(
        "/api/v1/search",
        json={"query": "test query", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "test query"
    assert "results" in data
    assert "search_time_ms" in data


def test_suggestions_endpoint():
    resp = client.get("/api/v1/suggestions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_preview_document_not_found():
    resp = client.get("/api/v1/documents/nonexistent/preview")
    assert resp.status_code == 404


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
