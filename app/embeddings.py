"""
Phase 2a — Turning text into vectors.

The one non-obvious concept here: RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY.

Gemini's embedding model supports "task types" - a hint about what the text
you're embedding IS. A document chunk being stored for later search and a
user's live question are different kinds of text (one is usually longer and
more declarative, the other shorter and more interrogative), so the model
embeds them slightly differently depending on which task_type you tell it.

This matters because retrieval is fundamentally asymmetric: you embed
chunks once at upload time with RETRIEVAL_DOCUMENT, and you embed each
question at search time with RETRIEVAL_QUERY. Using the same task_type for
both still technically works, but using the matched pair measurably
improves how well a question's vector lands near the right chunk's vector -
this is literally what the model was trained to optimize for.
"""
from __future__ import annotations

from google import genai
from google.genai import types

from app.config import settings

_client: genai.Client | None = None

# 768 keeps vectors small (fast to store/search) while still being one of
# Google's recommended sizes (768/1536/3072). Go up to 1536 or 3072 if you
# want marginally better retrieval accuracy at the cost of more storage.
EMBEDDING_DIMENSIONS = 768


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document chunks for storage. Use at ingestion time."""
    if not texts:
        return []
    client = get_client()
    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return [e.values for e in response.embeddings]


def embed_query(text: str) -> list[float]:
    """Embed a single user question for search. Use at search time."""
    client = get_client()
    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return response.embeddings[0].values