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
from app.models import DocumentChunk
from app.vectorstore import search

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


def _format_context(chunks: list[DocumentChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        location = f", page {c.page_number}" if c.page_number else ""
        parts.append(f"[Source {i}] (from {c.doc_name}{location})\n{c.text}")
    return "\n\n".join(parts)


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

    return {
        "answer": answer_text,
        "sources": [
            {
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "chunk_id": c.chunk_id,
                "excerpt": c.text[:200],
            }
            for c in chunks
        ],
    }