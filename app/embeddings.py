"""
Embedding layer — turns text into vectors via Google's Gemini API.

Key design decisions:
  - RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY task types (asymmetric embedding)
  - Retry with exponential backoff for transient API failures
  - Parallel embedding execution via ThreadPoolExecutor ensuring exact 1-to-1 chunk mapping
  - Lazy client initialization
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.exceptions import EmbeddingError
from app.logging_config import get_logger, log_duration

logger = get_logger(__name__)

_client: genai.Client | None = None

# 768 keeps vectors small (fast to store/search) while still being one of
# Google's recommended sizes (768/1536/3072).
EMBEDDING_DIMENSIONS = 768


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
def _embed_single(text: str, task_type: str) -> list[float]:
    """Embed a single text string with retry logic. Returns a 768-dim float list."""
    client = get_client()
    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    if not response.embeddings or not response.embeddings[0].values:
        raise EmbeddingError("Gemini API returned an empty embedding vector")
    return list(response.embeddings[0].values)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document chunks concurrently.
    Guarantees exactly one vector embedding per input text, maintaining order."""
    if not texts:
        return []

    with log_duration(logger, f"Embedding {len(texts)} document chunks"):
        embeddings: list[list[float] | None] = [None] * len(texts)

        # Use max 10 concurrent threads to avoid rate limits while maintaining high throughput
        max_workers = min(10, max(1, len(texts)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_embed_single, text, "RETRIEVAL_DOCUMENT"): i
                for i, text in enumerate(texts)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embeddings[index] = future.result()
                except Exception as e:
                    raise EmbeddingError(
                        f"Failed to embed chunk index {index} (length: {len(texts[index])})",
                        detail=str(e),
                    ) from e

        # Final safety validation
        result = [emb for emb in embeddings if emb is not None]
        if len(result) != len(texts):
            raise EmbeddingError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(result)}"
            )

        return result


def embed_query(text: str) -> list[float]:
    """Embed a single user question for search."""
    try:
        return _embed_single(text, "RETRIEVAL_QUERY")
    except Exception as e:
        raise EmbeddingError(
            "Failed to embed search query",
            detail=str(e),
        ) from e