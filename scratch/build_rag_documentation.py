"""
Script to generate RAG_Masterclass_Guide.pdf and RAG_Masterclass_Guide.md
Comprehensive Enterprise RAG Architecture & Implementation Blueprint
"""
import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "RAG_Masterclass_Guide.pdf"
MD_PATH = BASE_DIR / "RAG_Masterclass_Guide.md"

print(f"Generating documentation in: {BASE_DIR}")

# ─── MARKDOWN CONTENT GENERATION ──────────────────────────────────────────────

md_content = """# 📚 RAG Masterclass & Architecture Blueprint
## Enterprise Retrieval-Augmented Generation with AetherMind AI

---

## 📖 Table of Contents
1. [Introduction to Retrieval-Augmented Generation (RAG)](#1-introduction-to-retrieval-augmented-generation-rag)
2. [Core Concepts & Fundamentals](#2-core-concepts--fundamentals)
   - [2.1 Why RAG? (RAG vs Fine-Tuning vs Prompting)](#21-why-rag-rag-vs-fine-tuning-vs-prompting)
   - [2.2 Document Extraction & Normalization](#22-document-extraction--normalization)
   - [2.3 Text Chunking Strategies & Overlap Math](#23-text-chunking-strategies--overlap-math)
   - [2.4 Dense Embeddings & Vector Spaces](#24-dense-embeddings--vector-spaces)
   - [2.5 Asymmetric Embeddings (Document vs Query)](#25-asymmetric-embeddings-document-vs-query)
   - [2.6 Vector Stores & Similarity Search (Cosine, L2, HNSW)](#26-vector-stores--similarity-search-cosine-l2-hnsw)
   - [2.7 Query Rewriting & Multi-Turn Context Memory](#27-query-rewriting--multi-turn-context-memory)
   - [2.8 Grounding, Citations & Hallucination Guardrails](#28-grounding-citations--hallucination-guardrails)
   - [2.9 Real-Time Asynchronous Streaming & Thread Pools](#29-real-time-asynchronous-streaming--thread-pools)
3. [System Architecture & Data Flow](#3-system-architecture--data-flow)
4. [File-by-File Technical Deep Dive](#4-file-by-file-technical-deep-dive)
   - [`app/config.py` — Centralized Configuration Management](#appconfigpy--centralized-configuration-management)
   - [`app/exceptions.py` — Application Error Hierarchy](#appexceptionspy--application-error-hierarchy)
   - [`app/logging_config.py` — Structured Logging & Performance Tracking](#applogging_configpy--structured-logging--performance-tracking)
   - [`app/models.py` — Data Schemas & Domain Enums](#appmodelspy--data-schemas--domain-enums)
   - [`app/ingestion.py` — Multi-Format Parsing & Deduplication Engine](#appingestionpy--multi-format-parsing--deduplication-engine)
   - [`app/embeddings.py` — Parallel Embedding Pipeline](#appembeddingspy--parallel-embedding-pipeline)
   - [`app/vectorstore.py` — ChromaDB Layer & KB Analytics](#appvectorstorepy--chromadb-layer--kb-analytics)
   - [`app/chat_memory.py` — SQLite Persistence, Rename & Export](#appchat_memorypy--sqlite-persistence-rename--export)
   - [`app/rag.py` — Grounded Generator, Prompts & Suggestions](#appragpy--grounded-generator-prompts--suggestions)
   - [`app/main.py` — FastAPI REST API & Streaming Gateway](#appmainpy--fastapi-rest-api--streaming-gateway)
   - [`static/index.html` — Glassmorphism UI & Knowledge Explorer](#staticindexhtml--glassmorphism-ui--knowledge-explorer)
5. [Advanced Topics & Production Optimization](#5-advanced-topics--production-optimization)
6. [Summary & Key Takeaways](#6-summary--key-takeaways)

---

## 1. Introduction to Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (**RAG**) is an artificial intelligence architecture that combines the power of **Large Language Models (LLMs)** with external information retrieval systems. Standard LLMs (such as Google Gemini, GPT-4, or Claude) are trained on vast datasets, but their knowledge is limited by two critical constraints:

1. **Cutoff Dates**: They do not possess knowledge of real-time or recent events past their training cutoff.
2. **Private Data Inaccessibility**: They have no access to your organization's internal documents, PDFs, private reports, or personal databases.

Trying to solve this by simply pasting massive documents directly into prompt windows causes high latency, immense token costs, and attention degradation ("lost in the middle" phenomenon).

**RAG solves this elegantly**: Instead of feeding entire documents into the LLM, RAG indexes the documents into compact numeric representations called **vector embeddings**. When a user asks a question, the system retrieves only the top 3-5 most relevant passages from the documents and feeds *only those specific passages* to the LLM alongside the question.

---

## 2. Core Concepts & Fundamentals

### 2.1 Why RAG? (RAG vs Fine-Tuning vs Prompting)

| Technique | How it Works | Primary Use Case | Cost & Maintenance | Updates | Grounding / Citations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Context Prompting** | Pasting context directly into prompt | One-off small inputs (<5k words) | Cheap per query, degrades at scale | Manual | Low |
| **Fine-Tuning** | Re-training model weights on custom dataset | Changing tone, style, or syntax | Expensive, requires GPUs & data | Hard (requires re-training) | None (Hallucinates easily) |
| **RAG** | Dynamically retrieving relevant passages | Document QA, KB Search, Enterprise Search | Low cost, highly scalable | Instant (just upload/delete doc) | **100% Verifiable Citations** |

---

### 2.2 Document Extraction & Normalization

Before text can be searched, raw files must be converted into clean, standardized plain text.

1. **PDF Extraction (`pypdf`)**: Reads PDF page objects, extracts string elements per page, and preserves page-level metadata (`page_number`).
2. **DOCX Extraction (`python-docx`)**: Iterates through structural document paragraphs and extracts non-empty text runs.
3. **TXT & Markdown**: Parsed directly using UTF-8 text decoding.
4. **Text Normalization**: Control characters (non-printable ASCII) are stripped, runs of 3+ newlines are collapsed to double newlines, and consecutive spaces are normalized. This ensures uniform embedding quality without whitespace artifacts.

---

### 2.3 Text Chunking Strategies & Overlap Math

LLMs and embedding models have input token boundaries. Feeding a 100-page document as a single string destroys search precision. RAG divides documents into smaller units called **chunks**.

- **Chunk Size (`CHUNK_SIZE_WORDS = 300`)**: The number of words in each chunk window. 300 words (~400 tokens) provides enough context for a coherent topic without diluting search signals.
- **Chunk Overlap (`CHUNK_OVERLAP_WORDS = 50`)**: The number of words duplicated between adjacent chunks.

#### Why Chunk Overlap is Essential
Without overlap, critical information spanning across chunk boundaries gets split in half. For instance, if sentence A is in Chunk 1 and sentence B (which completes the thought) is in Chunk 2, neither chunk alone contains the full idea. Overlapping by 50 words ensures boundary context is preserved in both chunks.

---

### 2.4 Dense Embeddings & Vector Spaces

An **embedding** is a mathematical transformation that turns text into a high-dimensional vector of floating-point numbers (e.g., 768 floats):

`"Financial quarterly report"` $\rightarrow$ `[0.0142, -0.0891, 0.4120, ..., 0.0051]`

Semantically similar phrases map to points that are geometrically close in vector space:
- `"The company earned $5M in revenue"` and `"Quarterly profits reached five million dollars"` will have vectors positioned close together, even if they share zero common words.

---

### 2.5 Asymmetric Embeddings (Document vs Query)

Google Gemini's `gemini-embedding-001` supports **asymmetric task types**:

1. **`RETRIEVAL_DOCUMENT`**: Optimized for long document passages stored in the vector database.
2. **`RETRIEVAL_QUERY`**: Optimized for short user questions.

Because questions are structured differently than statements ("What was Q3 profit?" vs "Q3 profit was $4.2M"), training separate projection heads for document indexing vs query searching yields significantly higher search accuracy.

---

### 2.6 Vector Stores & Similarity Search (Cosine, L2, HNSW)

A **Vector Store** (such as ChromaDB) is a specialized database built for indexing high-dimensional vectors and conducting nearest-neighbor searches.

- **L2 Distance (Euclidean)**: Measures straight-line distance between two vector endpoints: $d(u,v) = \sqrt{\sum (u_i - v_i)^2}$.
- **Cosine Similarity**: Measures the cosine of the angle between two vectors: $\text{similarity} = \frac{u \cdot v}{\|u\| \|v\|}$.
- **HNSW Index (Hierarchical Navigable Small World)**: A graph-based indexing algorithm that allows sub-linear $\mathcal{O}(\log N)$ similarity searches across millions of vectors without performing brute-force comparisons.

Our similarity conversion formula:
$$\text{Similarity Score} = \frac{1.0}{1.0 + \text{L2 Distance}}$$

---

### 2.7 Query Rewriting & Multi-Turn Context Memory

In multi-turn chat conversations, users naturally ask follow-up questions containing pronouns:
- Turn 1: *"Tell me about project AetherMind."*
- Turn 2: *"When was it launched?"*

If we directly embed *"When was it launched?"*, vector search will search for generic launch events because `"it"` is ambiguous.

**Query Rewriting**: Before vector retrieval, an LLM pass analyzes the recent conversation history and rewrites the follow-up:
`"When was it launched?"` $\rightarrow$ `"When was project AetherMind launched?"`

This standalone question is then embedded and searched, delivering accurate context retrieval.

---

### 2.8 Grounding, Citations & Hallucination Guardrails

To eliminate hallucinations, AetherMind AI implements strict grounding rules:

1. **Strict System Prompt**: Instructs the model to answer *ONLY* using the retrieved context blocks and decline to answer if the context does not contain the information.
2. **Inline Citations**: Every fact is tagged with `[Source N]` markers.
3. **Grounding Safety Heuristic (`check_grounding`)**: Scans assistant responses to verify that either:
   - The response explicitly contains `[Source N]` citation markers, OR
   - The response contains standard refusal phrases (e.g., *"The documents do not contain..."*).
   If an answer lacks both citations and refusal markers, it is flagged with a warning badge.

---

### 2.9 Real-Time Asynchronous Streaming & Thread Pools

To stream responses token-by-token without blocking the FastAPI event loop:

1. **Producer-Consumer Queue (`asyncio.Queue`)**: A background worker thread executes the blocking Gemini streaming API generator and places tokens into an `asyncio.Queue`.
2. **Async Generator**: The FastAPI endpoint yields tokens asynchronously to the client using **NDJSON (Newline-Delimited JSON)**.
3. **Parallel Embedding Engine (`ThreadPoolExecutor`)**: Concurrent embedding of document chunks across up to 10 worker threads, ensuring 100% deterministic 1-to-1 matching between input chunks and returned vectors.

---

## 3. System Architecture & Data Flow

```mermaid
flowchart TD
    User([User / Web UI]) -->|HTTP / Streaming NDJSON| API[FastAPI Gateway app/main.py]

    subgraph Ingestion Pipeline
        API -->|Upload Document| Extract[ingestion.py: Extract & Normalize]
        Extract --> Chunk[ingestion.py: Overlapping Chunking]
        Chunk --> EmbedParallel[embeddings.py: ThreadPoolExecutor]
        EmbedParallel --> ChromaDB[(vectorstore.py: ChromaDB)]
    end

    subgraph RAG Retrieval & Generation
        API -->|Chat Query| Rewrite[rag.py: Query Rewriting]
        Rewrite --> EmbedQuery[embeddings.py: RETRIEVAL_QUERY]
        EmbedQuery --> VectorSearch[vectorstore.py: Cosine Search]
        VectorSearch --> Prompt[rag.py: Context Prompt Assembly]
        Prompt --> Gemini[(Google Gemini API)]
        Gemini --> SafetyCheck[rag.py: Grounding Safety Heuristic]
        SafetyCheck --> API
    end

    API <--> SQLite[(chat_memory.py: SQLite DB)]
```

---

## 4. File-by-File Technical Deep Dive

### `app/config.py` — Centralized Configuration Management
- **Purpose**: Single source of truth for configuration variables loaded from `.env` via `pydantic-settings`.
- **Key Components**:
  - `Settings`: Schema defining `gemini_api_key`, `gemini_model`, `embedding_model`, `chunk_size_words`, `chunk_overlap_words`, `top_k_results`, `relevance_threshold`.
  - `@field_validator`: Validates that `chunk_overlap_words < chunk_size_words`.
  - `validate_startup()`: Warns at server launch if mandatory API keys or configurations are missing.

---

### `app/exceptions.py` — Application Error Hierarchy
- **Purpose**: Defines custom exception classes inheriting from `RAGBaseError`.
- **Classes**:
  - `RAGBaseError`: Base class carrying a human-readable `detail` message.
  - `IngestionError`: Raised during text parsing/chunking failures.
  - `EmbeddingError`: Raised when Gemini embedding API requests fail after retries.
  - `VectorStoreError`: Raised on database write/query errors.
  - `GenerationError`: Raised when LLM generation fails or returns empty responses.

---

### `app/logging_config.py` — Structured Logging & Performance Tracking
- **Purpose**: Provides colorized console output with rich tracebacks and operation timing utilities.
- **Key Components**:
  - `setup_logging()`: Initializes RichHandler logging.
  - `log_duration(logger, operation)`: Context manager measuring execution time in seconds.

---

### `app/models.py` — Data Schemas & Domain Enums
- **Purpose**: Pydantic models for request/response validation and OpenAPI documentation.
- **Key Models & Enums**:
  - `AnswerMode`: Enum (`DETAILED`, `CONCISE`, `BULLET_POINTS`).
  - `ExportFormat`: Enum (`MARKDOWN`, `JSON`).
  - `DocumentChunk`: Internal representation of a text window with chunk index, page number, and content hash.
  - `ChatRequest`, `ChatResponse`: Schemas for RAG chat interactions.
  - `SearchRequest`, `SearchResponse`: Schemas for the standalone semantic search endpoint.
  - `StatsResponse`: Comprehensive system statistics schema.

---

### `app/ingestion.py` — Multi-Format Parsing & Deduplication Engine
- **Purpose**: Extracts text from PDF, DOCX, TXT, MD files, normalizes whitespace, hashes content, and splits text into overlapping chunks.
- **Key Functions**:
  - `compute_file_hash(path)`: SHA-256 hash of raw file bytes for duplicate upload detection.
  - `extract_pages(path)`: Text extraction returning page-level tuples `(page_number, text)`.
  - `chunk_text(text, size, overlap)`: Overlapping word-count window generator.
  - `chunk_document(path)`: Full ingestion orchestration returning `(doc_id, list[DocumentChunk])`. Filters out blank/whitespace-only chunks.

---

### `app/embeddings.py` — Parallel Embedding Pipeline
- **Purpose**: Interacts with Google's `google-genai` SDK to convert text strings into 768-dimensional float vectors.
- **Key Functions**:
  - `_embed_single(text, task_type)`: Embeds a single text string with exponential backoff retries (`tenacity`).
  - `embed_documents(texts)`: Uses `ThreadPoolExecutor` (up to 10 workers) to embed chunks in parallel, guaranteeing an exact 1-to-1 match between input texts and vector embeddings.
  - `embed_query(text)`: Embeds search queries using `RETRIEVAL_QUERY`.

---

### `app/vectorstore.py` — ChromaDB Layer & KB Analytics
- **Purpose**: Manages storage, similarity search, document lifecycle, and knowledge base metrics via ChromaDB.
- **Key Functions**:
  - `add_chunks(chunks)`: Embeds and stores chunks with metadata in ChromaDB.
  - `search(query, top_k, doc_ids)`: Basic similarity search.
  - `search_with_scores(query, top_k, doc_ids)`: Similarity search returning `(DocumentChunk, relevance_score)` tuples.
  - `get_document_preview(doc_id)`: Fetches first N chunks for document preview modals.
  - `get_knowledge_base_summary()`: Computes total words, storage size, and chunk distribution.
  - `delete_document(doc_id)`: Deletes stored vector embeddings and unlinks the original file from disk.

---

### `app/chat_memory.py` — SQLite Persistence, Rename & Export
- **Purpose**: Persists chat sessions and message histories to a local SQLite database (`chat_sessions.db`).
- **Key Functions**:
  - `create_session()`, `get_session()`, `append_message()`: CRUD session operations.
  - `rename_session(session_id, title)`: Updates session titles.
  - `export_session(session_id, format)`: Formats and exports chat histories as Markdown or JSON.

---

### `app/rag.py` — Grounded Generator, Prompts & Suggestions
- **Purpose**: Core orchestration of query rewriting, context prompt assembly, LLM streaming, grounding checks, and prompt suggestions.
- **Key Functions**:
  - `rewrite_query_async(history, question)`: Resolves conversation context into standalone search queries.
  - `answer_question_stream_async(...)`: Async producer-consumer streaming generator using `asyncio.Queue`.
  - `check_grounding(answer, num_sources)`: Evaluates citation presence vs refusal phrases.
  - `generate_suggestions(doc_names)`: Generates smart contextual prompt suggestions based on uploaded document names.

---

### `app/main.py` — FastAPI REST API & Streaming Gateway
- **Purpose**: Mounts HTTP endpoints, security headers, rate limiters (`slowapi`), CORS, and static UI files.
- **Key Endpoints**:
  - `POST /api/v1/documents/upload`: Uploads and ingests files.
  - `GET /api/v1/documents/{doc_id}/preview`: Preview document chunks.
  - `POST /api/v1/search`: Standalone semantic search.
  - `POST /api/v1/chat/stream`: Async token-by-token RAG streaming (NDJSON).
  - `PATCH /api/v1/sessions/{session_id}/rename`: Rename chat session.
  - `GET /api/v1/sessions/{session_id}/export`: Export chat history.
  - `GET /api/v1/suggestions`: Prompt suggestions.
  - `GET /api/v1/stats`: System metrics.

---

### `static/index.html` — Glassmorphism UI & Knowledge Explorer
- **Purpose**: Modern ChatGPT-style single-page frontend.
- **Features**:
  - Glassmorphism dark mode with animated mesh background.
  - Dual Views: **Chat View** (streaming QA) and **Knowledge Explorer View** (semantic chunk search).
  - Modal dialogs for Document Preview and System Settings.
  - Inline double-click session renaming.
  - Message actions (copy text, copy code, regenerate).
  - Keyboard shortcuts (`Ctrl+N`, `Ctrl+K`, `Escape`).

---

## 5. Advanced Topics & Production Optimization

1. **Hybrid Search (Dense + Sparse/BM25)**: Combining vector semantic search with BM25 keyword search for acronyms or product codes.
2. **Re-Ranking (Cross-Encoders)**: Using a re-ranker model (e.g., Cohere Rerank) to re-order top 20 initial vector results before prompt assembly.
3. **Semantic Chunking**: Splitting documents by semantic sentence similarity rather than fixed word counts.
4. **Vector Database Scaling**: Transitioning from local ChromaDB to distributed vector stores (Qdrant, Milvus, or Pinecone) for multi-million document scaling.

---

## 6. Summary & Key Takeaways

- **RAG eliminates hallucinations** by forcing LLMs to answer strictly from retrieved document facts.
- **Overlapping chunking** prevents context loss across window boundaries.
- **Asymmetric embeddings** optimize document storage vs query matching.
- **Parallel embedding** ensures 1-to-1 chunk-to-vector mapping and eliminates length mismatches.
- **Async producer-consumer queues** deliver real-time streaming without blocking event loops.

---
*Created for AetherMind AI — Enterprise Knowledge Platform Blueprint*
"""

# Write Markdown file
with open(MD_PATH, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"[SUCCESS] Markdown Guide created: {MD_PATH}")

# ─── REPORTLAB PDF GENERATION ──────────────────────────────────────────────────

styles = getSampleStyleSheet()

# Custom styles
primary_color = colors.HexColor("#7c3aed")  # Deep Purple
secondary_color = colors.HexColor("#06b6d4")  # Cyan
dark_text = colors.HexColor("#1e293b")  # Dark Slate
light_bg = colors.HexColor("#f8fafc")  # Light slate bg
border_color = colors.HexColor("#e2e8f0")

title_style = ParagraphStyle(
    "CoverTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=26,
    leading=32,
    textColor=primary_color,
    alignment=0,  # Left align
    spaceAfter=10,
)

subtitle_style = ParagraphStyle(
    "CoverSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=14,
    leading=18,
    textColor=secondary_color,
    alignment=0,
    spaceAfter=25,
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    textColor=primary_color,
    spaceBefore=18,
    spaceAfter=8,
    keepWithNext=True,
)

h2_style = ParagraphStyle(
    "SectionH2",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=17,
    textColor=colors.HexColor("#0f172a"),
    spaceBefore=14,
    spaceAfter=6,
    keepWithNext=True,
)

h3_style = ParagraphStyle(
    "SectionH3",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=15,
    textColor=colors.HexColor("#334155"),
    spaceBefore=10,
    spaceAfter=4,
    keepWithNext=True,
)

body_style = ParagraphStyle(
    "BodyTextCustom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=14.5,
    textColor=dark_text,
    spaceAfter=8,
)

bullet_style = ParagraphStyle(
    "BulletCustom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=14,
    textColor=dark_text,
    leftIndent=15,
    firstLineIndent=-10,
    spaceAfter=4,
)

code_style = ParagraphStyle(
    "CodeSnippet",
    parent=styles["Normal"],
    fontName="Courier",
    fontSize=8.5,
    leading=11,
    textColor=colors.HexColor("#0f172a"),
    backColor=colors.HexColor("#f1f5f9"),
    borderColor=colors.HexColor("#cbd5e1"),
    borderWidth=0.5,
    borderPadding=6,
    spaceBefore=6,
    spaceAfter=8,
    borderRadius=4,
)

callout_style = ParagraphStyle(
    "CalloutText",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=9.5,
    leading=14,
    textColor=colors.HexColor("#1e1b4b"),
    spaceAfter=0,
)


def make_callout(text: str):
    p = Paragraph(f"💡 <b>Key Takeaway:</b> {text}", callout_style)
    t = Table([[p]], colWidths=[6.5 * inch])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f3ff")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ddd6fe")),
            ("LINELEFT", (0, 0), (0, -1), 3.5, primary_color),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ])
    )
    return t


doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=letter,
    leftMargin=0.75 * inch,
    rightMargin=0.75 * inch,
    topMargin=0.75 * inch,
    bottomMargin=0.75 * inch,
)

story = []

# Title Banner
story.append(Paragraph("AetherMind AI — RAG Masterclass", title_style))
story.append(
    Paragraph("Complete Enterprise Architecture, Concepts & Implementation Manual", subtitle_style)
)
story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=15))

# Intro
story.append(Paragraph("1. Introduction to Retrieval-Augmented Generation", h1_style))
story.append(
    Paragraph(
        "Retrieval-Augmented Generation (<b>RAG</b>) is an architecture that couples large language models with external information retrieval systems. Standard LLMs are constrained by training cutoffs and lack access to private enterprise data. RAG addresses this by indexing documents into vector space and fetching relevant excerpts dynamically during chat queries.",
        body_style,
    )
)

story.append(make_callout("RAG provides 100% grounded answers with exact source citations without needing to re-train or fine-tune LLM model weights."))
story.append(Spacer(1, 10))

# Core Concepts
story.append(Paragraph("2. Core Concepts & Fundamentals", h1_style))

story.append(Paragraph("2.1 Document Extraction & Normalization", h2_style))
story.append(
    Paragraph(
        "Raw PDF, DOCX, TXT, and Markdown files are ingested, stripped of non-printable control characters, and normalized. Double-newlines separate paragraphs while keeping page numbers intact for citation tracking.",
        body_style,
    )
)

story.append(Paragraph("2.2 Text Chunking & Overlap Math", h2_style))
story.append(
    Paragraph(
        "Documents are split into overlapping word windows (e.g., 300 words with 50 words overlap). Overlap ensures that thoughts spanning across chunk boundaries are not lost in vector space.",
        body_style,
    )
)
story.append(Paragraph("<b>Overlap Math Formula:</b> Step Size = Chunk Size - Overlap Size", bullet_style))
story.append(Paragraph("Example: 300 size - 50 overlap = 250 word step between window starts.", bullet_style))

story.append(Paragraph("2.3 Vector Embeddings & Asymmetric Search", h2_style))
story.append(
    Paragraph(
        "Embeddings map text into 768-dimensional float vectors. Asymmetric task types (<b>RETRIEVAL_DOCUMENT</b> vs <b>RETRIEVAL_QUERY</b>) optimize the vector projection for storage vs query matching.",
        body_style,
    )
)

story.append(Paragraph("2.4 Similarity Search & HNSW Indexing", h2_style))
story.append(
    Paragraph(
        "ChromaDB computes L2 Euclidean distances between vectors. We convert L2 distance into a normalized 0–1 similarity score: <b>Similarity = 1.0 / (1.0 + L2_Distance)</b>.",
        body_style,
    )
)

story.append(Paragraph("2.5 Query Rewriting & Multi-Turn Memory", h2_style))
story.append(
    Paragraph(
        "Context-dependent questions (e.g., 'When was it launched?') are automatically rewritten into standalone search queries ('When was project AetherMind launched?') before embedding lookup.",
        body_style,
    )
)

story.append(Paragraph("2.6 Grounding & Safety Heuristics", h2_style))
story.append(
    Paragraph(
        "Strict prompt guardrails force Gemini to answer strictly from retrieved context and append [Source N] tags. Grounding safety checks inspect generated responses for citation compliance.",
        body_style,
    )
)

story.append(Spacer(1, 10))

# Architecture Table
story.append(Paragraph("3. File-by-File Technical Deep Dive", h1_style))

table_data = [
    [
        Paragraph("<b>File Path</b>", body_style),
        Paragraph("<b>Core Purpose</b>", body_style),
        Paragraph("<b>Key Technologies</b>", body_style),
    ],
    [
        Paragraph("<code>app/config.py</code>", code_style),
        Paragraph("Centralized settings & startup checks", body_style),
        Paragraph("Pydantic BaseSettings, dotenv", body_style),
    ],
    [
        Paragraph("<code>app/exceptions.py</code>", code_style),
        Paragraph("Custom application exception hierarchy", body_style),
        Paragraph("RAGBaseError base class", body_style),
    ],
    [
        Paragraph("<code>app/logging_config.py</code>", code_style),
        Paragraph("Structured logging & timing context manager", body_style),
        Paragraph("Rich console, duration logger", body_style),
    ],
    [
        Paragraph("<code>app/models.py</code>", code_style),
        Paragraph("Domain data models & API schemas", body_style),
        Paragraph("Pydantic V2, Enums", body_style),
    ],
    [
        Paragraph("<code>app/ingestion.py</code>", code_style),
        Paragraph("Parsing, SHA-256 dedup, window chunking", body_style),
        Paragraph("pypdf, python-docx, hashlib", body_style),
    ],
    [
        Paragraph("<code>app/embeddings.py</code>", code_style),
        Paragraph("Parallel 1-to-1 chunk embedding pipeline", body_style),
        Paragraph("google-genai, ThreadPoolExecutor", body_style),
    ],
    [
        Paragraph("<code>app/vectorstore.py</code>", code_style),
        Paragraph("ChromaDB vector store & KB analytics", body_style),
        Paragraph("ChromaDB, PersistentClient", body_style),
    ],
    [
        Paragraph("<code>app/chat_memory.py</code>", code_style),
        Paragraph("SQLite chat history, title & export", body_style),
        Paragraph("SQLite3, Markdown export", body_style),
    ],
    [
        Paragraph("<code>app/rag.py</code>", code_style),
        Paragraph("Query rewriting, streaming RAG, grounding", body_style),
        Paragraph("asyncio.Queue, Gemini 2.5 Flash", body_style),
    ],
    [
        Paragraph("<code>app/main.py</code>", code_style),
        Paragraph("FastAPI REST endpoints & NDJSON stream", body_style),
        Paragraph("FastAPI, SlowAPI, CORS", body_style),
    ],
    [
        Paragraph("<code>static/index.html</code>", code_style),
        Paragraph("Glassmorphism UI & Knowledge Explorer", body_style),
        Paragraph("Vanilla JS, Prism.js, Marked.js", body_style),
    ],
]

t_files = Table(table_data, colWidths=[1.8 * inch, 2.7 * inch, 2.0 * inch])
t_files.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])
)

story.append(t_files)
story.append(Spacer(1, 15))

# Detailed File Descriptions
story.append(Paragraph("4. Key Implementation Highlights", h1_style))

story.append(Paragraph("Parallel Embedding Pipeline (`app/embeddings.py`)", h2_style))
story.append(
    Paragraph(
        "To prevent vector count mismatch errors when indexing large PDF files, <code>embed_documents()</code> executes individual chunk embeddings concurrently using <code>ThreadPoolExecutor</code>. This guarantees exact 1-to-1 matching between input chunks and returned 768-dimensional vector embeddings while offering high throughput.",
        body_style,
    )
)

story.append(Paragraph("Async Token Streaming (`app/rag.py` & `app/main.py`)", h2_style))
story.append(
    Paragraph(
        "Token-by-token streaming is achieved by running Gemini's blocking iterator in a producer thread that feeds tokens into an <code>asyncio.Queue</code>. The FastAPI endpoint consumes from the queue asynchronously, streaming NDJSON lines to the browser without blocking the main event loop.",
        body_style,
    )
)

story.append(Spacer(1, 15))
story.append(make_callout("AetherMind AI v2.0.0 represents a complete production-ready RAG platform with unit test coverage across all layers."))

# Build PDF
doc.build(story)

print(f"[SUCCESS] PDF Masterclass Guide created: {PDF_PATH}")
