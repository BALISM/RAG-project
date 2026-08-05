"""
RAG pipeline — retrieval + grounded generation with source citations.

Pipeline:
  1. Embed the question (RETRIEVAL_QUERY)
  2. Search the vector store for top_k most similar chunks
  3. Format those chunks as numbered context
  4. Ask Gemini to answer USING ONLY that context
  5. Check grounding and return answer with citations

Async architecture:
  - All blocking Gemini API calls are wrapped in asyncio.to_thread so they
    never block the event loop, allowing true streaming to the browser.
"""
from __future__ import annotations

import asyncio
import queue
import threading

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


async def rewrite_query_async(history: list[ChatMessage], question: str) -> str:
    """Async wrapper around rewrite_query using a background thread."""
    return await asyncio.to_thread(rewrite_query, history, question)


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
    """Synchronous streaming version (kept for reference / non-async contexts)."""
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


async def answer_question_stream_async(
    question: str,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
):
    """
    Async streaming version of the RAG pipeline.

    Uses a producer thread + asyncio.Queue to bridge the blocking
    Gemini generate_content_stream iterator into the async event loop.
    This ensures the event loop is NEVER blocked, giving true real-time
    token streaming to the browser with zero buffering delay.
    """
    # Step 1: Vector search in background thread (non-blocking)
    yield {"type": "status", "text": "Searching knowledge base..."}
    chunks = await asyncio.to_thread(
        lambda: search(question, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)
    )

    if not chunks:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": _NO_DOCUMENTS_MSG}
        yield {"type": "done", "text": _NO_DOCUMENTS_MSG, "grounded": True, "warning": None}
        return

    yield {"type": "sources", "sources": _build_sources(chunks)}

    context = _format_context(chunks)
    prompt = _RAG_PROMPT.format(context=context, question=question)

    yield {"type": "status", "text": "Generating answer..."}

    # Step 2: Run the blocking Gemini streaming call in a background thread,
    # bridging tokens into an asyncio.Queue so we can await them here.
    token_queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()  # signals that the stream is finished

    def _run_stream():
        """Blocking producer: runs in a thread, puts tokens into the queue."""
        try:
            client = get_client()
            for event in client.models.generate_content_stream(
                model=settings.gemini_model,
                contents=prompt,
            ):
                piece = getattr(event, "text", None)
                if piece:
                    # Schedule putting the token into the queue from the main loop
                    loop.call_soon_threadsafe(token_queue.put_nowait, piece)
        except Exception as exc:
            loop.call_soon_threadsafe(token_queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(token_queue.put_nowait, _SENTINEL)

    loop = asyncio.get_event_loop()
    thread = threading.Thread(target=_run_stream, daemon=True)
    thread.start()

    full_text = ""
    while True:
        item = await token_queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise GenerationError(f"Streaming failed: {item}") from item
        full_text += item
        yield {"type": "token", "text": item}

    if not full_text:
        raise GenerationError("Empty streamed response from Gemini")

    grounding = check_grounding(full_text, len(chunks))
    yield {"type": "done", "text": full_text, **grounding}