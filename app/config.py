"""Centralized config, same pattern as the YT project - one place that reads
.env, everything else imports `settings` from here."""
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    embedding_model: str = "text-embedding-001"

    chunk_size_words: int = 300
    chunk_overlap_words: int = 50
    top_k_results: int = 4

    upload_dir: str = "uploads"
    chroma_dir: str = "chroma_db"

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


settings = Settings()