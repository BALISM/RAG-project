# RAG Chatbot

Upload documents, then ask questions grounded in exactly what's in them.
Built phase by phase to actually learn how RAG (Retrieval-Augmented
Generation) works, not just wire an existing library together.
POST /documents/upload Upload a .pdf/.docx/.txt/.md file
GET /documents List uploaded documents
GET /documents/{doc_id} Full detail for one document, including its chunks
DELETE /documents/{doc_id} Remove a document (vectors AND the file on disk)
POST /chat Ask a question, get a grounded answer back
POST /chat/stream Same, but streamed token-by-token
A browser UI is served at `/` — upload files, check specific ones to scope
a question, and chat with streaming responses and source citations.

---

## Project structure
A browser UI is served at `/` — upload files, check specific ones to scope
a question, and chat with streaming responses and source citations.

---

## Project structure
app/
config.py Settings, loaded from .env
models.py All Pydantic schemas (chunks, chat messages/sessions, API I/O)
ingestion.py Phase 1 — text extraction (PDF/DOCX/TXT) + overlapping chunking
embeddings.py Phase 2 — Gemini embedding calls (RETRIEVAL_DOCUMENT/QUERY)
vectorstore.py Phase 2/5 — Chroma wrapper: storage, search, dedup, delete
rag.py Phase 4/6 — retrieval + generation, query rewriting, grounding check
chat_memory.py Phase 6 — in-memory chat session store
main.py FastAPI routes tying everything together
static/index.html Browser UI: upload, document list, streaming chat
tests/
conftest.py Shared fixtures (isolated vector store, fake embeddings)
test_ingestion.py Extraction, hashing, chunking (18 tests)
test_vectorstore.py Storage, search, dedup, delete (18 tests)