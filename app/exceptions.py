"""
Custom exception hierarchy for the RAG Chatbot.

All application-specific exceptions inherit from RAGBaseError, making it easy
to catch 'any app error' at the API layer while still differentiating between
ingestion failures, embedding failures, vector store issues, and generation
problems in more targeted handlers.
"""
from __future__ import annotations


class RAGBaseError(Exception):
    """Base for all RAG Chatbot application errors."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail or message


class IngestionError(RAGBaseError):
    """Raised when document extraction or chunking fails."""


class EmbeddingError(RAGBaseError):
    """Raised when the embedding API call fails after retries."""


class VectorStoreError(RAGBaseError):
    """Raised when ChromaDB operations fail."""


class GenerationError(RAGBaseError):
    """Raised when the LLM generation call fails."""


class SessionError(RAGBaseError):
    """Raised for chat session management issues."""
