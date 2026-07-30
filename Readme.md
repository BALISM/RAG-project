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
---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)

uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

For the test suite: `pip install -r requirements-dev.txt && pytest tests/ -v`

**⚠️ Never commit a real API key.** `.gitignore` excludes `.env`, but
`.env.example` should only ever contain the placeholder text
`your_gemini_api_key_here` — this project has already had one real key
accidentally committed via a filled-in `.env.example` on a public repo,
which Google's own leak-detection bots found within hours. If that
happens: revoke the key immediately at the link above, generate a new one,
and don't rely on `git rm` alone — old commits still contain the key until
the history itself is scrubbed.

---

## How RAG actually works here (the concepts, not just the code)

**Chunking exists for a different reason than in a pure summarization
pipeline.** In a "summarize this whole document" task, chunking only needs
to make things *fit* inside one API call. Here, chunking determines
*search precision* — each chunk is independently embedded and independently
searchable, so a chunk that's too big buries the relevant sentence in noise,
and a chunk that's too small loses surrounding context. `ingestion.py` uses
overlapping word-count windows (`CHUNK_SIZE_WORDS`/`CHUNK_OVERLAP_WORDS`) so
a sentence landing on a chunk boundary still ends up complete in at least
one chunk instead of being cut in half in both.

**Embeddings are asymmetric.** `embeddings.py` embeds document chunks with
`task_type="RETRIEVAL_DOCUMENT"` at upload time, and embeds questions with
`task_type="RETRIEVAL_QUERY"` at search time. These aren't interchangeable —
Gemini's embedding model was trained to place a well-matched
question/answer pair close together in vector space *specifically when*
each side is tagged with its correct role.

**Retrieval and generation are two separate steps, done fresh every time.**
Unlike ingestion (which happens once per document, at upload time),
`rag.py`'s `answer_question()` re-embeds the question, re-searches the
vector store, and re-generates an answer on every single call. Nothing
about a question is cached or reused across turns except the chat history
itself.

**Grounding is enforced by the prompt, then checked afterward.** The RAG
prompt in `rag.py` explicitly forbids answering from outside knowledge and
requires citing `[Source N]` markers. `check_grounding()` is a cheap
heuristic safety net on top of that instruction — it flags any answer that
neither cites a source nor admits the information is missing, since that
combination usually means the model quietly ignored the instruction. It's
not a real fact-checker (it can't tell if a cited source was represented
*accurately*), just a catch for the most common and most dangerous failure
mode: a confident, uncited answer from general knowledge.

**Query rewriting happens before retrieval, not after.** A raw follow-up
like *"what about his education?"* means nothing to an embedding model in
isolation — "his" has no referent. `rewrite_query()` in `rag.py` uses the
conversation history to turn it into a standalone question *before* it's
ever embedded or searched. This only costs an extra API call on turn 2+ of
a conversation — the very first question in any session is standalone by
definition and skips this step entirely.

---

## Feature map (what got built, in order)

- **Phase 1 — Ingestion**: PDF/DOCX/TXT/MD text extraction with per-page
  tracking (PDFs only), overlapping word-count chunking, SHA256 content
  hashing for later dedup.
- **Phase 2 — Embeddings + vector store**: Gemini embeddings via
  `gemini-embedding-001`, stored in a local persistent Chroma database.
- **Phase 3 — API**: upload/list/delete endpoints wiring ingestion and
  storage together.
- **Phase 4 — RAG itself**: retrieval + grounded generation with source
  citations.
- **Phase 5 — Multi-document handling**: duplicate-upload detection by
  content hash (re-uploading the same file reuses the existing entry
  instead of creating a duplicate), per-document detail view, search
  scoped to a specific set of documents instead of the whole library,
  delete that actually removes the file from disk (not just the vectors).
- **Phase 6 — Chat memory**: session-based conversation history, query
  rewriting for context-dependent follow-ups.
- **Phase 7 — Polish**: full browser UI (upload, document selection,
  streaming chat with citations), streaming responses end to end, a
  grounding/hallucination heuristic check, file size limits and a document
  count cap enforced against real bytes/counts rather than trusted
  client-provided values.
- **Testing**: 36 automated tests across ingestion and vector store
  behavior, all running offline against fake embeddings (no API key or
  network needed to run the suite).

---

## Configuration reference (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | from https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | `gemini-3.6-flash` | used for rewriting, generation, and the grounding-adjacent prompts |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | `text-embedding-004` was retired Jan 2026 - don't use it |
| `CHUNK_SIZE_WORDS` | `300` | target words per chunk |
| `CHUNK_OVERLAP_WORDS` | `50` | words duplicated between consecutive chunks |
| `TOP_K_RESULTS` | `4` | chunks retrieved per question |
| `MAX_FILE_SIZE_MB` | `20` | enforced against actual bytes written, not headers |
| `MAX_DOCUMENTS` | `50` | cap on distinct documents; dedup reuse bypasses this |

---

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

36 tests, fully offline — `conftest.py` provides an `isolated_chroma`
fixture (fresh temp vector store per test, no shared state) and a
`fake_embeddings` fixture (deterministic stand-in vectors, no real API
calls). Every test file was validated from a completely clean virtualenv
with only the pinned `requirements.txt`/`requirements-dev.txt` versions
before being handed off, to make sure nothing depends on leftover local
state.

What's **not** covered by the automated suite, since it genuinely needs a
live API key and network access: real embedding quality, real Gemini
generation quality, and the streaming endpoint against a real (not faked)
`generate_content_stream` response. Test those by actually running the app
and uploading a real document.

---

## Known limitations

- **Chat memory is in-memory only** — restarting the server loses every
  conversation's history (the vector store persists on disk separately and
  survives restarts fine; only chat sessions don't).
- **No auth or rate limiting** — fine for local/personal use, add before
  exposing this publicly.
- **The grounding check is a heuristic, not a fact-checker** — see above.
  It catches "answered with no citation and no admission of a gap," not
  "cited a source but misrepresented what it says."
- **No semantic/embedding-based chunk-boundary detection** — chunking is
  still fixed word-count windows with overlap, not topic-aware splitting.
- **Whisper-style audio ingestion isn't supported** — only text-extractable
  PDF/DOCX/TXT/MD. A scanned, image-only PDF will fail extraction with a
  clear error rather than silently returning nothing.