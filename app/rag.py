"""
Phase 4 — Retrieval + Generation. This is "RAG" itself, finally.

The pipeline, in order:
  1. Embed the question (RETRIEVAL_QUERY - see embeddings.py)
  2. Search the vector store for the top_k most similar chunks
  3. Stuff those chunks into a prompt as "context"
  4. Ask Gemini to answer USING ONLY that context - explicitly told not to
     use outside knowledge, and explicitly told to say so if the answer
     isn't in the provided context
  5. Return the answer AND which chunks it came from (citations)

Step 4's instruction is the whole trust model of RAG: without it, the LLM
will happily answer from its own general knowledge even when your documents
say nothing relevant, and you can no longer tell the difference between "the
model answered from your data" and "the model guessed." Citations in step 5
are what make that trust checkable rather than assumed.
"""
from __future__ import annotations

from google.genai import types

from app.config import settings
from app.embeddings import get_client
from app.models import ChatMessage, DocumentChunk
from app.vectorstore import search

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
    """Turns a context-dependent follow-up ("what about his education?")
    into something that means something on its own ("what is Muhammad
    Balaj's educational background?") BEFORE it gets embedded and
    searched. Skipped entirely when there's no history yet - the first
    question in a conversation is standalone by definition, so there's
    nothing to rewrite and no reason to spend an extra API call on it."""
    if not history:
        return question

    # Last 3 turns (6 messages) is plenty of context for resolving a
    # pronoun or a "what about X" - older history rarely matters for THIS
    # specific job and just costs more tokens for no benefit.
    recent = history[-6:]
    history_text = "\n".join(f"{m.role}: {m.content}" for m in recent)
    prompt = _REWRITE_PROMPT.format(history=history_text, question=question)

    client = get_client()
    response = client.models.generate_content(model=settings.gemini_model, contents=prompt)
    rewritten = getattr(response, "text", None)

    # If rewriting fails for any reason, fall back to the original question
    # rather than blocking the whole chat - a slightly-worse-quality search
    # beats a hard error on a non-essential step.
    return rewritten.strip() if rewritten and rewritten.strip() else question

_RAG_PROMPT = """\
Answer the question using ONLY the context below, which was retrieved from \
the user's uploaded documents. If the answer isn't contained in the \
context, say clearly that the documents don't contain that information - \
do NOT use any outside knowledge to fill the gap, even if you know the \
answer generally.

When you use a piece of context, mention which source it came from using \
the [Source N] markers already present in the context below, so the reader \
knows exactly where each claim is grounded.

Context:
{context}

Question: {question}
"""


class RagError(Exception):
    pass


# Phrases that indicate the model correctly declined to answer rather than
# guessing - these count as "grounded" even with zero citations, because
# admitting "it's not in the documents" IS the correct grounded behavior,
# not a failure.
_REFUSAL_PHRASES = (
    "don't contain", "doesn't contain", "do not contain", "does not contain",
    "isn't in the", "is not in the", "isn't contained", "is not contained",
    "no information", "not mentioned", "not provided", "not available in",
)


def check_grounding(answer: str, num_sources: int) -> dict:
    """A cheap heuristic safety net, not a real hallucination detector -
    a genuine one would need a separate LLM call to verify each claim
    against the source text, which costs money and latency on every single
    answer. This instead checks for the one specific failure mode that
    matters most: an answer that neither cites anything NOR admits the
    information isn't there. That combination almost always means the
    model quietly fell back to outside knowledge instead of the provided
    context - exactly what the RAG prompt tells it not to do."""
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
            "This answer doesn't cite any source and doesn't say the information "
            "is missing either - it may not be fully grounded in your documents."
        ),
    }


def _format_context(chunks: list[DocumentChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        location = f", page {c.page_number}" if c.page_number else ""
        parts.append(f"[Source {i}] (from {c.doc_name}{location})\n{c.text}")
    return "\n\n".join(parts)


def _build_sources(chunks: list[DocumentChunk]) -> list[dict]:
    return [
        {
            "doc_name": c.doc_name,
            "page_number": c.page_number,
            "chunk_id": c.chunk_id,
            "excerpt": c.text[:200],
        }
        for c in chunks
    ]


def answer_question(
    question: str,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
) -> dict:
    """The full Phase 4 pipeline. Returns {"answer": str, "sources": [...]}"""
    chunks = search(question, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)

    if not chunks:
        return {
            "answer": "No documents have been uploaded yet, so there's nothing to search.",
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
        raise RagError("Empty response from Gemini")

    grounding = check_grounding(answer_text, len(chunks))
    return {"answer": answer_text, "sources": _build_sources(chunks), **grounding}


def answer_question_stream(
    question: str,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
):
    """Same retrieval + prompt as answer_question, but the generation step
    streams tokens as they arrive instead of waiting for the whole answer.
    Yields dicts, always in this order:
      1. one {"type": "sources", "sources": [...]} - fired BEFORE any text,
         so a UI can show citations immediately rather than waiting for the
         full answer to finish (retrieval already happened; there's no
         reason to withhold that from the caller just because generation
         hasn't finished yet)
      2. repeated {"type": "token", "text": "..."} as each piece arrives
      3. one final {"type": "done", "text": "<full answer>"} - so callers
         don't have to manually concatenate every token themselves
    """
    chunks = search(question, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)

    if not chunks:
        message = "No documents have been uploaded yet, so there's nothing to search."
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": message}
        yield {"type": "done", "text": message, "grounded": True, "warning": None}
        return

    yield {"type": "sources", "sources": _build_sources(chunks)}

    context = _format_context(chunks)
    prompt = _RAG_PROMPT.format(context=context, question=question)

    client = get_client()
    full_text = ""
    for event in client.models.generate_content_stream(model=settings.gemini_model, contents=prompt):
        piece = getattr(event, "text", None)
        if piece:
            full_text += piece
            yield {"type": "token", "text": piece}

    if not full_text:
        raise RagError("Empty streamed response from Gemini")

    grounding = check_grounding(full_text, len(chunks))
    yield {"type": "done", "text": full_text, **grounding}