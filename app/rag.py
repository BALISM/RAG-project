"""
RAG pipeline — retrieval + grounded generation with source citations.

Pipeline:
  1. Embed the question (RETRIEVAL_QUERY)
  2. Search the vector store for top_k most similar chunks
  3. Format those chunks as numbered context
  4. Ask Gemini to answer USING ONLY that context
  5. Check grounding and return answer with citations

Enhanced:
  - Multi-mode answer prompts (Detailed / Concise / Bullet Points)
  - Performance metrics (retrieval time, generation time, source count)
  - Enhanced grounding checks with citation-count validation
  - Smart suggestion generation from document content

Async architecture:
  - All blocking Gemini API calls are wrapped in asyncio.to_thread so they
    never block the event loop, allowing true streaming to the browser.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time

from google.genai import types

from app.config import settings
from app.embeddings import get_client
from app.exceptions import GenerationError
from app.logging_config import get_logger, log_duration
from app.models import AnswerMode, ChatMessage, DocumentChunk
from app.vectorstore import search, search_with_scores

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


# ─── Multi-Mode RAG Prompts ─────────────────────────────────────────────────

_RAG_PROMPTS = {
    AnswerMode.DETAILED: """\
You are a helpful assistant that answers questions based ONLY on the \
provided context from the user's uploaded documents.

## Rules
1. Answer ONLY from the context below. Do NOT use outside knowledge.
2. If the answer is not in the context, say clearly that the documents \
don't contain that information.
3. Cite your sources using [Source N] markers matching the context labels.
4. Be thorough and comprehensive. Provide detailed explanations with examples from the documents.
5. Use markdown formatting for clarity — headers, bold, lists, etc.
6. If multiple sources discuss the topic, synthesize them coherently.

## Context
{context}

## Question
{question}

## Answer""",

    AnswerMode.CONCISE: """\
You are a helpful assistant that answers questions based ONLY on the \
provided context from the user's uploaded documents.

## Rules
1. Answer ONLY from the context below. Do NOT use outside knowledge.
2. If the answer is not in the context, say clearly that the documents \
don't contain that information.
3. Cite your sources using [Source N] markers matching the context labels.
4. Be CONCISE — aim for 2-4 sentences maximum. Get straight to the point.
5. Use markdown formatting sparingly.

## Context
{context}

## Question
{question}

## Answer""",

    AnswerMode.BULLET_POINTS: """\
You are a helpful assistant that answers questions based ONLY on the \
provided context from the user's uploaded documents.

## Rules
1. Answer ONLY from the context below. Do NOT use outside knowledge.
2. If the answer is not in the context, say clearly that the documents \
don't contain that information.
3. Cite your sources using [Source N] markers matching the context labels.
4. Format your answer as a **bullet-point list**. Each bullet should be one key point.
5. Include a brief 1-sentence summary at the top before the bullets.

## Context
{context}

## Question
{question}

## Answer""",
}

# Legacy fallback
_RAG_PROMPT = _RAG_PROMPTS[AnswerMode.DETAILED]


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
    admit the information is missing — the most common hallucination pattern.

    Enhanced: also checks that the number of cited sources is reasonable
    relative to the number of source chunks provided.
    """
    if num_sources == 0:
        return {"grounded": True, "warning": None}

    answer_lower = answer.lower()
    has_citation = "[source" in answer_lower
    is_refusal = any(phrase in answer_lower for phrase in _REFUSAL_PHRASES)

    if has_citation or is_refusal:
        # Additional check: count citations vs available sources
        cited_count = answer_lower.count("[source")
        if has_citation and cited_count > num_sources * 3:
            return {
                "grounded": False,
                "warning": (
                    f"The answer contains {cited_count} citation markers but only "
                    f"{num_sources} sources were provided — some citations may be fabricated."
                ),
            }
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


def _build_sources(chunks: list[DocumentChunk], scores: list[float] | None = None) -> list[dict]:
    sources = []
    for i, c in enumerate(chunks):
        source = {
            "source_index": i + 1,
            "doc_name": c.doc_name,
            "page_number": c.page_number,
            "chunk_id": c.chunk_id,
            "excerpt": c.text[:200],
        }
        if scores and i < len(scores):
            source["relevance_score"] = round(scores[i], 4)
        sources.append(source)
    return sources


# ─── Generation ──────────────────────────────────────────────────────────────

_NO_DOCUMENTS_MSG = "No documents have been uploaded yet, so there's nothing to search. Please upload a document first."
_NO_RELEVANT_MSG = "I searched your documents but couldn't find any passages relevant enough to answer this question. Try rephrasing your question or uploading additional documents."


def answer_question(
    question: str,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
    answer_mode: AnswerMode = AnswerMode.DETAILED,
) -> dict:
    """The full RAG pipeline.  Returns answer, sources, grounding status, and metrics."""
    metrics = {}
    t_start = time.perf_counter()

    with log_duration(logger, f"Answering: '{question[:60]}'"):
        # Retrieval with scores
        t_retrieval = time.perf_counter()
        scored_chunks = search_with_scores(question, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)
        metrics["retrieval_time_ms"] = round((time.perf_counter() - t_retrieval) * 1000, 1)

        if not scored_chunks:
            return {
                "answer": _NO_RELEVANT_MSG,
                "sources": [],
                "grounded": True,
                "warning": None,
                "metrics": metrics,
            }

        chunks = [pair[0] for pair in scored_chunks]
        scores = [pair[1] for pair in scored_chunks]

        context = _format_context(chunks)
        prompt_template = _RAG_PROMPTS.get(answer_mode, _RAG_PROMPT)
        prompt = prompt_template.format(context=context, question=question)

        t_generation = time.perf_counter()
        client = get_client()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        metrics["generation_time_ms"] = round((time.perf_counter() - t_generation) * 1000, 1)

        answer_text = getattr(response, "text", None)
        if not answer_text:
            raise GenerationError("Empty response from Gemini")

        grounding = check_grounding(answer_text, len(chunks))
        metrics["total_time_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        metrics["num_sources"] = len(chunks)

        return {
            "answer": answer_text,
            "sources": _build_sources(chunks, scores),
            **grounding,
            "metrics": metrics,
        }


def answer_question_stream(
    question: str,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
    answer_mode: AnswerMode = AnswerMode.DETAILED,
):
    """Synchronous streaming version (kept for reference / non-async contexts)."""
    yield {"type": "status", "text": "Searching knowledge base..."}
    scored_chunks = search_with_scores(question, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)

    if not scored_chunks:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": _NO_RELEVANT_MSG}
        yield {"type": "done", "text": _NO_RELEVANT_MSG, "grounded": True, "warning": None}
        return

    chunks = [pair[0] for pair in scored_chunks]
    scores = [pair[1] for pair in scored_chunks]

    yield {"type": "sources", "sources": _build_sources(chunks, scores)}

    context = _format_context(chunks)
    prompt_template = _RAG_PROMPTS.get(answer_mode, _RAG_PROMPT)
    prompt = prompt_template.format(context=context, question=question)

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
    answer_mode: AnswerMode = AnswerMode.DETAILED,
):
    """
    Async streaming version of the RAG pipeline.

    Uses a producer thread + asyncio.Queue to bridge the blocking
    Gemini generate_content_stream iterator into the async event loop.
    This ensures the event loop is NEVER blocked, giving true real-time
    token streaming to the browser with zero buffering delay.
    """
    t_start = time.perf_counter()

    # Step 1: Vector search in background thread (non-blocking)
    yield {"type": "status", "text": "Searching knowledge base..."}
    t_retrieval = time.perf_counter()
    scored_chunks = await asyncio.to_thread(
        lambda: search_with_scores(question, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)
    )
    retrieval_ms = round((time.perf_counter() - t_retrieval) * 1000, 1)

    if not scored_chunks:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": _NO_RELEVANT_MSG}
        yield {"type": "done", "text": _NO_RELEVANT_MSG, "grounded": True, "warning": None, "metrics": {"retrieval_time_ms": retrieval_ms}}
        return

    chunks = [pair[0] for pair in scored_chunks]
    scores = [pair[1] for pair in scored_chunks]

    yield {"type": "sources", "sources": _build_sources(chunks, scores)}

    context = _format_context(chunks)
    prompt_template = _RAG_PROMPTS.get(answer_mode, _RAG_PROMPT)
    prompt = prompt_template.format(context=context, question=question)

    yield {"type": "status", "text": "Generating answer..."}

    # Step 2: Run the blocking Gemini streaming call in a background thread,
    # bridging tokens into an asyncio.Queue so we can await them here.
    token_queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()  # signals that the stream is finished

    t_generation = time.perf_counter()

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

    generation_ms = round((time.perf_counter() - t_generation) * 1000, 1)

    if not full_text:
        raise GenerationError("Empty streamed response from Gemini")

    grounding = check_grounding(full_text, len(chunks))
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    yield {
        "type": "done",
        "text": full_text,
        **grounding,
        "metrics": {
            "retrieval_time_ms": retrieval_ms,
            "generation_time_ms": generation_ms,
            "total_time_ms": total_ms,
            "num_sources": len(chunks),
        },
    }


# ─── New: Smart Suggestions ─────────────────────────────────────────────────


def generate_suggestions(doc_names: list[str]) -> list[dict]:
    """Generate smart prompt suggestions based on uploaded document names.
    Returns a list of suggestion dicts with title and question fields."""
    if not doc_names:
        return [
            {"title": "Getting Started", "description": "Upload a document first", "question": "What documents do I have?"},
        ]

    docs_text = ", ".join(doc_names[:10])

    prompt = f"""\
Given these uploaded documents: {docs_text}

Generate exactly 4 smart prompt suggestions that a user might want to ask about these documents.
Each suggestion should have a short title (2-4 words) and a full question.

Return ONLY a JSON array of objects with "title", "description" (short 3-5 word desc), and "question" fields.
No markdown, no code fences, just the JSON array.
"""

    try:
        client = get_client()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if text:
            import json
            # Strip code fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()
            suggestions = json.loads(text)
            if isinstance(suggestions, list):
                return suggestions[:6]
    except Exception:
        logger.warning("Failed to generate smart suggestions", exc_info=True)

    # Fallback suggestions based on doc names
    return [
        {"title": "Executive Summary", "description": "Summarize key insights", "question": f"Summarize the key takeaways from {doc_names[0]}"},
        {"title": "Key Topics", "description": "Find main themes", "question": "What are the main topics discussed across all documents?"},
        {"title": "Compare Documents", "description": "Cross-reference content", "question": "What common themes appear across the uploaded documents?"},
        {"title": "Deep Dive", "description": "Detailed analysis", "question": f"Give me a detailed analysis of the most important points in {doc_names[0]}"},
    ]