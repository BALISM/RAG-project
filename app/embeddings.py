"""
Embedding layer — turns text into vectors via Google's Gemini API.

Key design decisions:
  - RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY task types (asymmetric embedding)
  - Retry with exponential backoff for transient API failures
  - Batch size limiting to stay within API constraints
  - Lazy client initialization
"""
from __future__ import annotations

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings
from app.exceptions import EmbeddingError
from app.logging_config import get_logger, log_duration

logger = get_logger(__name__)

_client: genai.Client | None = None

# 768 keeps vectors small (fast to store/search) while still being one of
# Google's recommended sizes (768/1536/3072).
EMBEDDING_DIMENSIONS = 768

# Gemini embedding API accepts at most ~100 texts per batch call.
_MAX_BATCH_SIZE = 100


def get_client() -> genai.Client:
    """Lazy-initialize the Gemini client singleton."""
    global _client
    if _client is None:
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            raise EmbeddingError("GEMINI_API_KEY is not configured — set it in .env")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """Call the embedding API for a single batch, with retry logic."""
    client = get_client()
    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return [e.values for e in response.embeddings]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document chunks for storage. Automatically splits
    into sub-batches if the input exceeds the API's per-call limit."""
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    with log_duration(logger, f"Embedding {len(texts)} document chunks"):
        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[i : i + _MAX_BATCH_SIZE]
            try:
                embeddings = _embed_batch(batch, "RETRIEVAL_DOCUMENT")
                all_embeddings.extend(embeddings)
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to embed document batch ({len(batch)} texts)",
                    detail=str(e),
                ) from e

    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single user question for search."""
    try:
        result = _embed_batch([text], "RETRIEVAL_QUERY")
        return result[0]
    except Exception as e:
        raise EmbeddingError(
            "Failed to embed search query",
            detail=str(e),
        ) from e