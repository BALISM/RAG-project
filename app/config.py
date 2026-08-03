"""Centralized configuration loaded from .env via pydantic-settings.

Every configurable value lives here.  The rest of the app imports
``settings`` and never reads environment variables directly.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── LLM & Embeddings ────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size_words: int = 300
    chunk_overlap_words: int = 50

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k_results: int = 5
    relevance_threshold: float = 0.35

    # ── Limits ───────────────────────────────────────────────────────────
    max_file_size_mb: int = 20
    max_documents: int = 50

    # ── Rate Limiting ────────────────────────────────────────────────────
    rate_limit_chat: str = "20/minute"
    rate_limit_upload: str = "10/minute"

    # ── Storage Paths ────────────────────────────────────────────────────
    upload_dir: str = "uploads"
    chroma_dir: str = "chroma_db"
    sessions_db: str = "chat_sessions.db"

    # ── Server ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    cors_origins: str = "*"

    # ── Derived ──────────────────────────────────────────────────────────

    @field_validator("chunk_overlap_words")
    @classmethod
    def overlap_must_be_less_than_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size_words", 300)
        if v >= size:
            raise ValueError(f"chunk_overlap_words ({v}) must be < chunk_size_words ({size})")
        return v

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        p = BASE_DIR / self.upload_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        p = BASE_DIR / self.chroma_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def sessions_db_path(self) -> Path:
        p = BASE_DIR / self.sessions_db
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def validate_startup(self) -> list[str]:
        """Return a list of warnings/errors found during startup validation."""
        issues: list[str] = []
        if not self.gemini_api_key or self.gemini_api_key == "your_gemini_api_key_here":
            issues.append("GEMINI_API_KEY is not set — embedding and chat will fail")
        if self.chunk_overlap_words >= self.chunk_size_words:
            issues.append(
                f"CHUNK_OVERLAP_WORDS ({self.chunk_overlap_words}) must be < "
                f"CHUNK_SIZE_WORDS ({self.chunk_size_words})"
            )
        return issues


settings = Settings()