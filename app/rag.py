"""
RAG pipeline — retrieval + grounded generation with source citations.

Pipeline:
  1. Embed the question (RETRIEVAL_QUERY)
  2. Search the vector store for top_k most similar chunks
  3. Format those chunks as numbered context
  4. Ask Gemini to answer USING ONLY that context
  5. Check grounding and return answer with citations

Key improvements:
  - Improved RAG prompt with structured instructions
  - Response metadata (model used, retrieval count)
  - Configurable temperature
  - Better grounding check with refusal phrase detection
"""
from __future__ import annotations

from google.genai import types

from app.config import settings
from app.embeddings import get_client
from app.exceptions import GenerationError
from app.logging_config import get_logger, log_duration
from app.models import ChatMessage, DocumentChunk
from app.vectorstore import search

logger = get_logger(__name__)

# ─── Query Rewriting ──────────────────────────────────────────────────────────

_REWRITE_PROMPT = """\
Given the conversation so far and a new follow-up question, rewrite the \
follow-up as a standalone question that makes complete sense on its own, \
with no missing context (resolve pronouns like "he/it/that" into the \
actual thing they refer to, using the conversation history).

If the follow-up question is already standalone and doesn't depend on \
anything earlier in the conversation, return it completely unchanged.

Return ONLY the rewritten question, nothing else - no preamble, no quotes.

Conversation so far:
{history}

Follow-up question: {question}

Standalone question:"""


def rewrite_query(history: list[ChatMessage], question: str) -> str:
    """Turn a context-dependent follow-up into a standalone question
    BEFORE it gets embedded and searched.  Skipped when there's no
    history — the first question is standalone by definition."""
    if not history:
        return question

    # Last 3 turns (6 messages) is enough context for resolving references
    recent = history[-6:]
    history_text = "\n".join(f"{m.role}: {m.content}" for m in recent)
    prompt = _REWRITE_PROMPT.format(history=history_text, question=question)

    try:
        client = get_client()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        rewritten = getattr(response, "text", None)
        if rewritten and rewritten.strip():
            logger.info("Rewrote query: '%s' → '%s'", question[:60], rewritten.strip()[:60])
            return rewritten.strip()
    except Exception:
        logger.warning("Query rewriting failed, using original question", exc_info=True)

    return question


# ─── RAG Prompt ───────────────────────────────────────────────────────────────

_RAG_PROMPT = """\
You are a helpful assistant that answers questions based ONLY on the \
provided context from the user's uploaded documents.

## Rules
1. Answer ONLY from the context below. Do NOT use outside knowledge.
2. If the answer is not in the context, say clearly that the documents \
don't contain that information.
3. Cite your sources using [Source N] markers matching the context labels.
4. Be thorough but concise. Use markdown formatting for clarity.
5. If multiple sources discuss the topic, synthesize them coherently.

## Context
{context}

## Question
{question}

## Answer"""


# ─── Grounding Check ─────────────────────────────────────────────────────────

# Phrases indicating the model correctly declined to answer
_REFUSAL_PHRASES = (
    "don't contain", "doesn't contain", "do not contain", "does not contain",
    "isn't in the", "is not in the", "isn't contained", "is not contained",
    "no information", "not mentioned", "not provided", "not available in",
    "cannot find", "could not find", "no relevant", "not discussed",
)


def check_grounding(answer: str, num_sources: int) -> dict:
    """Heuristic safety net: flags answers that neither cite sources nor
    admit the information is missing — the most common hallucination pattern."""
    if num_sources == 0:
        return {"grounded": True, "warning": None}

    answer_lower = answer.lower()
    has_citation = "[source" in answer_lower
    is_refusal = any(phrase in answer_lower for phrase in _REFUSAL_PHRASES)

    if has_citation or is_refusal:
        return {"grounded": True, "warning": None}

    return {
        "grounded": False,
        "warning": (
            "This answer doesn't cite any source and doesn't indicate the "
            "information is missing — it may not be fully grounded in your documents."
        ),
    }


# ─── Context Formatting ──────────────────────────────────────────────────────


def _format_context(chunks: list[DocumentChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        location = f", page {c.page_number}" if c.page_number else ""
        parts.append(f"[Source {i}] (from {c.doc_name}{location})\n{c.text}")
    return "\n\n".join(parts)


def _build_sources(chunks: list[DocumentChunk]) -> list[dict]:
    return [
        {
            "source_index": i + 1,
            "doc_name": c.doc_name,
            "page_number": c.page_number,
            "chunk_id": c.chunk_id,
            "excerpt": c.text[:200],
        }
        for i, c in enumerate(chunks)
    ]


# ─── Generation ──────────────────────────────────────────────────────────────

_NO_DOCUMENTS_MSG = "No documents have been uploaded yet, so there's nothing to search. Please upload a document first."


def answer_question(
    question: str,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
) -> dict:
    """The full RAG pipeline.  Returns answer, sources, grounding status."""
    with log_duration(logger, f"Answering: '{question[:60]}'"):
        chunks = search(question, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)

        if not chunks:
            return {
                "answer": _NO_DOCUMENTS_MSG,
                "sources": [],
                "grounded": True,
                "warning": None,
            }

        context = _format_context(chunks)
        prompt = _RAG_PROMPT.format(context=context, question=question)

        client = get_client()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )

        answer_text = getattr(response, "text", None)
        if not answer_text:
            raise GenerationError("Empty response from Gemini")

        grounding = check_grounding(answer_text, len(chunks))
        return {"answer": answer_text, "sources": _build_sources(chunks), **grounding}


def answer_question_stream(
    question: str,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
):
    """Streaming version of answer_question.  Yields dicts:
      1. {"type": "sources", "sources": [...]}
      2. {"type": "token", "text": "..."}  (repeated)
      3. {"type": "done", "text": "<full>", "grounded": bool, "warning": ...}
    """
    yield {"type": "status", "text": "Searching knowledge base..."}
    chunks = search(question, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)

    if not chunks:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": _NO_DOCUMENTS_MSG}
        yield {"type": "done", "text": _NO_DOCUMENTS_MSG, "grounded": True, "warning": None}
        return

    yield {"type": "sources", "sources": _build_sources(chunks)}

    context = _format_context(chunks)
    prompt = _RAG_PROMPT.format(context=context, question=question)

    yield {"type": "status", "text": "Generating answer..."}

    client = get_client()
    full_text = ""
    for event in client.models.generate_content_stream(
        model=settings.gemini_model,
        contents=prompt,
    ):
        piece = getattr(event, "text", None)
        if piece:
            full_text += piece
            yield {"type": "token", "text": piece}

    if not full_text:
        raise GenerationError("Empty streamed response from Gemini")

    grounding = check_grounding(full_text, len(chunks))
    yield {"type": "done", "text": full_text, **grounding}