# Nexia — Wikipedia-Grounded AI Knowledge Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6.svg?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17_%2B_pgvector-4169E1.svg?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Gemma_3_4B-white.svg?style=flat&logo=ollama&logoColor=black)](https://ollama.com/)
[![Tests](https://img.shields.io/badge/Tests-100%25_Passing_(62%2F62)-brightgreen.svg?style=flat)](run_tests.py)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Nexia** is a locally hosted, production-grade, Wikipedia-grounded conversational AI agent. Engineered from first principles, Nexia pairs hybrid document retrieval (BM25 lexical search + dense pgvector embeddings) with a locally served **Gemma 3 4B** large language model running on Ollama. Nexia delivers verifiable, hallucination-resistant answers with complete evidence traceability, rigorous citation management, and a fluid, streaming React chat interface.

---

## Author & Project Identity

- **Original Author & Lead Engineer**: Deekshith1983
- **Contact Email**: [dikideeku@gmail.com](mailto:dikideeku@gmail.com)
- **Official GitHub Repository**: [https://github.com/Deekshith1983/Nexia](https://github.com/Deekshith1983/Nexia)

---

## Table of Contents

1. [Key Architectural Highlights](#key-architectural-highlights)
2. [End-to-End System Architecture](#end-to-end-system-architecture)
3. [Technology Stack](#technology-stack)
4. [Project Directory Layout](#project-directory-layout)
5. [Core Engineering Implementations](#core-engineering-implementations)
   - [Enforced Retrieval Configuration](#1-enforced-retrieval-configuration)
   - [Hybrid Retrieval & Slotted Evidence Packing](#2-hybrid-retrieval--slotted-evidence-packing)
   - [Inline Citation Stripping & Primary Source Footer](#3-inline-citation-stripping--primary-source-footer)
   - [SSE Streaming & Unified UX ("Thinking...")](#4-sse-streaming--unified-ux-thinking)
   - [Conversation Persistence & Dynamic Title Generation](#5-conversation-persistence--dynamic-title-generation)
   - [Admission Control, Rate Limiting & Concurrency](#6-admission-control-rate-limiting--concurrency)
6. [Prerequisites & System Requirements](#prerequisites--system-requirements)
7. [Installation & Setup](#installation--setup)
8. [Running the Application](#running-the-application)
9. [Automated Testing Suite](#automated-testing-suite)
10. [REST API Documentation](#rest-api-documentation)
11. [Configuration Reference](#configuration-reference)
12. [Contributing & Development Workflow](#contributing--development-workflow)
13. [License](#license)

---

## Key Architectural Highlights

- **100% Local Inference & Privacy**: Zero external LLM API dependency. Inference is powered by local Ollama serving Google's `gemma3:4b` and embeddings via `nomic-embed-text`.
- **Hybrid Retrieval Pipeline**: Combines BM25 Okapi lexical scoring with dense vector similarity via PostgreSQL `pgvector`, joined through dynamic reciprocal-rank fusion and slotted passage ranking.
- **Strict Evidence Grounding**: Hallucination suppression via strict contextual constraints. The model is permitted to formulate assertions only from verified Wikipedia passages.
- **Zero-Pollution Prose Generation**: Eliminates unsightly internal citation tags (`[k]`, `[1]`, `[source]`) during stream generation using regex lookbehind filters, appending a single primary Wikipedia hyperlinked citation footer at the close of synthesis.
- **Unified User Experience**: Clean, minimalist "Thinking..." status indicator without exposing internal pipeline state or technical retrieval jargon.
- **Hardware-Aware Admission Control**: Queue-bounded admission manager (`ResponseController`) prevents Ollama CPU saturation and memory thrashing on modest consumer hardware (e.g., Ryzen 5, 16GB RAM).
- **Consolidated Test Automation**: Single unified test runner (`run_tests.py`) verifying 62 unit and integration tests across backend and frontend with a 100% pass rate.

---

## End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Frontend["Frontend Client (React 19 + TypeScript + Vite)"]
        UI["Chat Interface (ChatWindow / MessageBubble)"]
        Sidebar["Conversation Sidebar (Virtual List)"]
        Store["State Management (Zustand: chatStore)"]
        StreamClient["SSE Stream Consumer (useChatStream)"]
    end

    subgraph Gateway["FastAPI Web Layer (Port 8000)"]
        CORSMiddleware["CORS & Request Middleware"]
        RateLimiter["TokenBucket Rate Limiter"]
        AdmissionMgr["ResponseController (Admission Queue)"]
        Routes_Conv["/api/v1/conversations"]
        Routes_Answer["/api/v1/retrieval/answer/stream"]
    end

    subgraph Services["Core Backend Services"]
        ConvMgr["ConversationManager (PostgreSQL + pgvector)"]
        AnswerOrch["AnswerOrchestrator"]
        ContextMgr["ContextBudgetManager (Sliding Window)"]
        CitationMgr["CitationManager (Single Primary Link)"]
        TitleGen["Dynamic TitleGenerator (Atomic Task)"]
    end

    subgraph RetrievalEngine["Staged Retrieval Engine"]
        WikiFetcher["Wikipedia Ingestion & Section Parser"]
        HybridSearch["Hybrid Searcher (BM25 + pgvector)"]
        EvidencePacker["Slotted Evidence Collector"]
    end

    subgraph LocalLLM["Local Ollama Service (Port 11434)"]
        Embedder["nomic-embed-text (768-dim)"]
        GemmaLLM["gemma3:4b (CPU-Optimized Inference)"]
    end

    %% Interactions
    UI -->|1. Submit Query| StreamClient
    StreamClient -->|2. POST /api/v1/retrieval/answer/stream| AdmissionMgr
    AdmissionMgr --> RateLimiter
    RateLimiter --> Routes_Answer
    Routes_Answer --> AnswerOrch

    AnswerOrch -->|Fetch History & Budget| ConvMgr
    AnswerOrch --> ContextMgr
    AnswerOrch -->|Retrieve Passages| WikiFetcher
    WikiFetcher --> HybridSearch
    HybridSearch -->|Dense Vector Query| Embedder
    HybridSearch --> EvidencePacker

    EvidencePacker -->|Pack Top Passages| AnswerOrch
    AnswerOrch -->|Prompt with Grounded Context| GemmaLLM
    GemmaLLM -->|Stream Raw Tokens| AnswerOrch
    AnswerOrch -->|Filter [k] Tokens & Clean Prose| CitationMgr
    CitationMgr -->|Append Primary Wikipedia Footer| Routes_Answer
    Routes_Answer -->|SSE Events: chunk, metadata, done| StreamClient
    StreamClient --> Store
    Store --> UI

    Routes_Conv --> ConvMgr
    ConvMgr -->|Trigger First Turn| TitleGen
    TitleGen --> GemmaLLM
    TitleGen -->|Atomic Update| ConvMgr
    Sidebar -->|Query Conversations| Routes_Conv
```

---

## Technology Stack

### Backend
- **Language**: Python 3.12+
- **Web Framework**: FastAPI 0.115+
- **ASGI Server**: Uvicorn with async event loops
- **Database ORM**: SQLAlchemy 2.0 (asyncio + asyncpg driver)
- **Vector Database**: PostgreSQL 17 + `pgvector` extension
- **Inference Server**: Ollama (local daemon)
- **Language Model**: Google `gemma3:4b`
- **Embedding Model**: `nomic-embed-text` (768 dimensions)
- **Information Retrieval**: `rank-bm25` (lexical) + cosine distance vector search (dense)
- **Validation & Serialization**: Pydantic v2
- **Testing**: `pytest`, `pytest-asyncio`, `httpx`

### Frontend
- **Framework**: React 19 + TypeScript
- **Build Tool**: Vite 6 (ESM bundler)
- **State Store**: Zustand (lightweight global state with selective subscriptions)
- **Server Cache & Sync**: TanStack Query v5
- **Styling**: Tailwind CSS + Shadcn/UI primitives
- **Virtualization**: `@tanstack/react-virtual` (for large chat and sidebar lists)
- **Markdown & Code**: `react-markdown`, `remark-gfm`, `rehype-highlight`
- **Icons**: Lucide React
- **Testing**: Vitest, React Testing Library, JSDOM

---

## Project Directory Layout

```text
Nexia/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── conversations.py       # Conversation management CRUD endpoints
│   │   │   │   ├── health.py              # Health check and telemetry routes
│   │   │   │   └── retrieval.py           # Synchronous & SSE streaming answer endpoints
│   │   │   └── dependencies.py            # FastAPI dependency injections
│   │   ├── core/
│   │   │   ├── config.py                  # Pydantic BaseSettings environment config
│   │   │   └── database.py                # Async SQLAlchemy engine & session maker
│   │   ├── models/
│   │   │   ├── conversation.py            # Conversation & Message database models
│   │   │   └── document.py                # Document, Section & Passage pgvector models
│   │   ├── retrieval/
│   │   │   ├── adaptive_nlp.py            # Query decomposition & intent classification
│   │   │   ├── evidence_collector.py      # Slotted passage packing & reranking
│   │   │   ├── hybrid_search.py           # BM25 + pgvector Reciprocal Rank Fusion
│   │   │   ├── passage_ranker.py          # Lexical/semantic passage similarity ranker
│   │   │   └── wiki_fetcher.py            # Wikipedia API client & section tree parser
│   │   ├── schemas/
│   │   │   ├── conversation.py            # Conversation DTOs & response contracts
│   │   │   └── retrieval.py               # Enforced retrieval configuration contracts
│   │   ├── services/
│   │   │   ├── answer_orchestrator.py     # Master coordinator for retrieval & generation
│   │   │   ├── cache_service.py           # In-memory and tiered caching layer
│   │   │   ├── citation_manager.py        # Single primary Wikipedia footer formatter
│   │   │   ├── context_budget_manager.py  # Sliding conversation memory token budgeting
│   │   │   ├── conversation_manager.py    # PostgreSQL repository for chat history
│   │   │   ├── llm_orchestrator.py        # Ollama client with streaming citation filter
│   │   │   ├── rate_limiter.py            # In-memory TokenBucket rate limiter
│   │   │   ├── response_controller.py     # Bounded queue inference admission manager
│   │   │   └── title_generator.py         # Autonomous LLM-based title summarizer
│   │   └── main.py                        # FastAPI application entrypoint & lifespan
│   ├── tests/
│   │   ├── conftest.py                    # Pytest async fixtures & DB engine disposal
│   │   └── test_nexia.py                  # Consolidated backend test suite (40 tests)
│   ├── .env.example                       # Backend configuration blueprint
│   ├── README.md                          # Backend-specific architecture guide
│   └── requirements.txt                   # Pinned backend dependencies
├── frontend/
│   ├── src/
│   │   ├── components/ui/                 # Shadcn/Tailwind reusable UI elements
│   │   ├── config/
│   │   │   └── retrieval.ts               # Shared RETRIEVAL_CONFIG enforcement constant
│   │   ├── features/
│   │   │   ├── chat/
│   │   │   │   ├── ChatWindow.tsx         # Main chat viewport with virtualization
│   │   │   │   ├── MessageBubble.tsx      # Clean markdown renderer & footer display
│   │   │   │   ├── ThinkingIndicator.tsx  # Unified "Thinking..." component
│   │   │   │   └── useChatStream.ts       # SSE event consumer hook
│   │   │   └── sidebar/
│   │   │       ├── ConversationItem.tsx   # Individual conversation row (select/edit/del)
│   │   │       ├── ConversationList.tsx   # Deduplicated sidebar list
│   │   │       └── Sidebar.tsx            # Left navigational collapsible sidebar
│   │   ├── services/
│   │   │   ├── api.ts                     # Centralized API client & payload builder
│   │   │   └── conversation.service.ts    # REST service for conversations
│   │   ├── store/
│   │   │   └── chatStore.ts               # Zustand store for messages, state & telemetry
│   │   └── test/
│   │       ├── setup.ts                   # Vitest matchers & mock setups
│   │       └── nexia.test.tsx             # Consolidated frontend test suite (22 tests)
│   ├── package.json                       # Frontend scripts & dependencies
│   ├── vite.config.ts                     # Bundler config & proxy routing
│   └── README.md                          # Frontend architecture guide
├── docs/
│   └── GITHUB_ISSUES.md                   # Prepared engineering backlog & roadmap issues
├── run_tests.py                           # Master unified test runner (Backend + Frontend)
├── .gitignore                             # Monorepo-wide ignore rules
└── README.md                              # Root comprehensive project manual
```

---

## Core Engineering Implementations

### 1. Enforced Retrieval Configuration

To guarantee predictable memory consumption and retrieval precision, both the backend Pydantic models (`backend/app/schemas/retrieval.py`) and the frontend API layer (`frontend/src/config/retrieval.ts`) strictly enforce the following retrieval contract:

```json
{
  "top_k_per_document": 3,
  "min_confidence": 0.4,
  "max_evidence_passages": 9
}
```

- **Frontend Enforcement**: `api.ts` merges `RETRIEVAL_CONFIG` *after* any caller arguments, preventing individual React components from overriding or degrading search parameters.
- **Backend Validation**: Pydantic schema validation sets these as defaults and rejects requests exceeding `max_evidence_passages = 9`.

### 2. Hybrid Retrieval & Slotted Evidence Packing

The retrieval engine (`backend/app/retrieval/`) operates through a four-stage pipeline:
1. **Wikipedia Acquisition**: Real-time Wikipedia page and section tree extraction via Wikipedia REST APIs (`wiki_fetcher.py`).
2. **Dense & Sparse Ingestion**: Parallel computation of sparse BM25 indices and 768-dimensional dense embeddings via `nomic-embed-text`.
3. **Reciprocal Rank Fusion (RRF)**: Merges rank positions across lexical and vector candidates to counter semantic drift.
4. **Slotted Evidence Collector** (`evidence_collector.py`): Allocates fixed passage slots (max 3 per document, up to 9 total) based on composite confidence scores, ensuring broad topical diversity without monopolization by a single verbose article.

### 3. Inline Citation Stripping & Primary Source Footer

- **Zero Inline Markers**: Local models frequently produce bracketed markers like `[k]`, `[1]`, or `[source]`. Nexia enforces clean prose generation via two complementary mechanisms:
  1. *Negative System Prompting*: Explicit instructions forbidding internal evidence markers.
  2. *StreamingCitationFilter* (`llm_orchestrator.py`): A regex-backed sliding window filter that buffers partial token streams, identifies, and strips bracketed markers on the fly before SSE dispatch.
- **Single Primary Source Footer**: At the conclusion of generation, `citation_manager.py` selects the single highest-scoring, verified Wikipedia article and appends a single markdown hyperlinked reference:
  ```markdown
  [Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein)
  ```
- No other citations appear in the body of the generated prose.

### 4. SSE Streaming & Unified UX ("Thinking...")

- **Server-Sent Events (`/api/v1/retrieval/answer/stream`)**: Dispatches real-time structured events:
  - `event: status` -> `"Thinking..."`
  - `event: chunk` -> `{"text": "..."}`
  - `event: metadata` -> `{"article_title": "...", "article_url": "..."}`
  - `event: done` -> Finalization signal
- **Unified UX**: Technical backend pipeline stages (e.g., "Extracting BM25 tokens", "Generating embeddings") are masked from the user. The UI displays exclusively `Thinking...` until the first token stream begins, adhering strictly to clean consumer UX standards.

### 5. Conversation Persistence & Dynamic Title Generation

- **PostgreSQL Persistence**: Complete thread history stored in `conversations` and `messages` tables with cascade deletion.
- **Dynamic Atomic Title Generation**: On the first turn of a new conversation, `title_generator.py` asynchronously generates a concise, 3-to-5 word title using Gemma 3 4B, updating the database atomically without blocking the primary streaming answer.
- **Context Budget Manager**: Employs a sliding-window token budgeter (`context_budget_manager.py`) to prune historical conversation turns when context nears model capacity (`LLM_NUM_CTX=6144`).

### 6. Admission Control, Rate Limiting & Concurrency

- **In-Memory Token Bucket** (`rate_limiter.py`): Protects answer generation and conversation CRUD against abuse.
- **ResponseController** (`response_controller.py`): Enforces an inference concurrency limit (`INFERENCE_MAX_ACTIVE=1`) with a bounded queue (`INFERENCE_MAX_QUEUE=3`). Additional requests wait up to 30 seconds or receive `HTTP 429 / 503` graceful backoff signals, preventing GPU/CPU memory crashes.

---

## Prerequisites & System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.12 or newer
- **Node.js**: Version 18.0 or newer (Node 20+ recommended)
- **PostgreSQL**: Version 16 or 17 with the `pgvector` extension installed
- **Ollama**: Local Ollama runtime installed and added to system PATH

### Pulling Ollama Models

Ensure Ollama is running (`ollama serve`), then pull the required models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Deekshith1983/Nexia.git
cd Nexia
```

### 2. Configure Backend Environment

Copy the blueprint configuration and adjust database credentials as needed:

```bash
cp backend/.env.example backend/.env
```

Ensure your PostgreSQL instance contains a database named `nexiav01` with the `vector` extension enabled:

```sql
CREATE DATABASE nexiav01;
\c nexiav01;
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Setup Python Virtual Environment

```bash
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

pip install -r backend/requirements.txt
```

### 4. Setup Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

### 1. Start the Backend Server

From the repository root (with virtual environment active):

```bash
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

- **Health Endpoint**: `http://localhost:8000/health`
- **Swagger Documentation**: `http://localhost:8000/docs`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

### 2. Start the Frontend Development Server

In a separate terminal:

```bash
cd frontend
npm run dev
```

- **Web Application**: Access `http://localhost:5173` in your browser.
- Vite automatically proxies `/api` and `/health` requests to `http://localhost:8000`.

### 3. Production Frontend Build

To build and verify the static assets:

```bash
npm --prefix frontend run build
```

---

## Automated Testing Suite

Nexia includes a consolidated test suite containing **62 tests** (40 backend, 22 frontend) spanning all functional, performance, and UI layers.

### Run All Tests via Unified Runner

Execute both backend and frontend test suites with formatted summary reporting:

```bash
python run_tests.py
```

### Run Test Suites Individually

**Backend Tests (`pytest`):**
```bash
pytest backend/tests/test_nexia.py -v
```

**Frontend Tests (`vitest`):**
```bash
npm --prefix frontend test
```

### Test Coverage Breakdown

| Layer | File | Tests | Coverage Areas |
| :--- | :--- | :--- | :--- |
| **Backend** | `backend/tests/test_nexia.py` | **40** | Health checks, metrics, API surface 404 validation, Conversation CRUD, Atomic title generation, Context budgeting, Adaptive NLP, Enforced retrieval config (3, 0.4, 9), Slotted evidence packing, Inline `[k]` marker stripping, Single primary citation footer, SSE streaming filter, TokenBucket rate limiter, ResponseController admission queue, In-memory cache. |
| **Frontend** | `frontend/src/test/nexia.test.tsx` | **22** | UI component rendering, Sidebar navigation, ConfirmModal, Zustand chat store, SSE stream reader, Unified "Thinking..." indicator, Enforced retrieval config (3, 0.4, 9), Error boundary, Virtualization container, Single Wikipedia hyperlinked citation rendering. |
| **Total** | | **62** | **100% Pass Rate** |

---

## REST API Documentation

### System Endpoints

| Method | Path | Description | Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service liveness probe | `{"status": "ok"}` |
| `GET` | `/metrics` | Operational performance metrics | Latency percentiles, cache hit ratios, queue depths |

### Conversation Management

| Method | Path | Description | Request Body |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/conversations` | List user conversations (paginated) | None |
| `POST` | `/api/v1/conversations` | Create a new conversation thread | `{"title": "Optional Title"}` |
| `GET` | `/api/v1/conversations/{id}` | Get conversation with messages | None |
| `PATCH` | `/api/v1/conversations/{id}` | Update conversation title | `{"title": "New Title"}` |
| `DELETE` | `/api/v1/conversations/{id}` | Delete conversation & messages | None |

### Retrieval & Answer Generation

| Method | Path | Description | Parameters / Payload |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/retrieval/answer` | Synchronous answer generation | `{"question": "string", "conversation_id": "uuid", "top_k_per_document": 3, "min_confidence": 0.4, "max_evidence_passages": 9}` |
| `POST` | `/api/v1/retrieval/answer/stream` | Server-Sent Events (SSE) streaming answer | Same payload as above |

---

## Configuration Reference

Key configuration settings managed in `backend/.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string with asyncpg driver |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API service host |
| `LLM_MODEL` | `gemma3:4b` | Ollama model identifier for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model identifier for dense embeddings |
| `EMBEDDING_DIMENSIONS` | `768` | Dimension vector size for embeddings |
| `INFERENCE_MAX_ACTIVE` | `1` | Max concurrent active LLM inferences |
| `INFERENCE_MAX_QUEUE` | `3` | Max requests queued before rejecting |
| `INFERENCE_QUEUE_TIMEOUT`| `30.0` | Max wait time (seconds) in admission queue |
| `RATE_LIMIT_ANSWER_CAPACITY` | `2` | Burst capacity for `/answer` endpoints |
| `RATE_LIMIT_ANSWER_REFILL_SECONDS` | `30.0` | Token refill period for `/answer` |

---

## Contributing & Development Workflow

Contributions are welcomed. Please follow this standard engineering workflow:

1. **Fork the Repository**: Create an isolated fork under your GitHub account.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Run Code Quality & Tests**:
   Ensure all 62 tests pass and frontend builds cleanly before committing:
   ```bash
   python run_tests.py
   npm --prefix frontend run build
   ```
4. **Adhere to Code Standards**:
   - Python: Follow PEP 8 and use type annotations throughout.
   - TypeScript: Maintain strict type safety; avoid `any`.
   - Documentation: Maintain docstrings and update README when modifying contracts.
5. **Open a Pull Request**: Submit your PR targeting `main` with a detailed explanation of changes and test evidence.

For planned improvements and backlog discussions, refer to [docs/GITHUB_ISSUES.md](docs/GITHUB_ISSUES.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <b>Nexia</b> — Designed & Maintained by <a href="https://github.com/Deekshith1983">Deekshith</a> (<a href="mailto:dikideeku@gmail.com">dikideeku@gmail.com</a>)
</p>

- **Web Application**: Access `http://localhost:5173` in your browser.
- Vite automatically proxies `/api` and `/health` requests to `http://localhost:8000`.

### 3. Production Frontend Build

To build and verify the static assets:

```bash
npm --prefix frontend run build
```

---

## Automated Testing Suite

Nexia includes a consolidated test suite containing **62 tests** (40 backend, 22 frontend) spanning all functional, performance, and UI layers.

### Run All Tests via Unified Runner

Execute both backend and frontend test suites with formatted summary reporting:

```bash
python run_tests.py
```

### Run Test Suites Individually

**Backend Tests (`pytest`):**
```bash
pytest backend/tests/test_nexia.py -v
```

**Frontend Tests (`vitest`):**
```bash
npm --prefix frontend test
```

### Test Coverage Breakdown

| Layer | File | Tests | Coverage Areas |
| :--- | :--- | :--- | :--- |
| **Backend** | `backend/tests/test_nexia.py` | **40** | Health checks, metrics, API surface 404 validation, Conversation CRUD, Atomic title generation, Context budgeting, Adaptive NLP, Enforced retrieval config (3, 0.4, 9), Slotted evidence packing, Inline `[k]` marker stripping, Single primary citation footer, SSE streaming filter, TokenBucket rate limiter, ResponseController admission queue, In-memory cache. |
| **Frontend** | `frontend/src/test/nexia.test.tsx` | **22** | UI component rendering, Sidebar navigation, ConfirmModal, Zustand chat store, SSE stream reader, Unified "Thinking..." indicator, Enforced retrieval config (3, 0.4, 9), Error boundary, Virtualization container, Single Wikipedia hyperlinked citation rendering. |
| **Total** | | **62** | **100% Pass Rate** |

---

## REST API Documentation

### System Endpoints

| Method | Path | Description | Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service liveness probe | `{"status": "ok"}` |
| `GET` | `/metrics` | Operational performance metrics | Latency percentiles, cache hit ratios, queue depths |

### Conversation Management

| Method | Path | Description | Request Body |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/conversations` | List user conversations (paginated) | None |
| `POST` | `/api/v1/conversations` | Create a new conversation thread | `{"title": "Optional Title"}` |
| `GET` | `/api/v1/conversations/{id}` | Get conversation with messages | None |
| `PATCH` | `/api/v1/conversations/{id}` | Update conversation title | `{"title": "New Title"}` |
| `DELETE` | `/api/v1/conversations/{id}` | Delete conversation & messages | None |

### Retrieval & Answer Generation

| Method | Path | Description | Parameters / Payload |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/retrieval/answer` | Synchronous answer generation | `{"question": "string", "conversation_id": "uuid", "top_k_per_document": 3, "min_confidence": 0.4, "max_evidence_passages": 9}` |
| `POST` | `/api/v1/retrieval/answer/stream` | Server-Sent Events (SSE) streaming answer | Same payload as above |

---

## Configuration Reference

Key configuration settings managed in `backend/.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string with asyncpg driver |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API service host |
| `LLM_MODEL` | `gemma3:4b` | Ollama model identifier for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model identifier for dense embeddings |
| `EMBEDDING_DIMENSIONS` | `768` | Dimension vector size for embeddings |
| `INFERENCE_MAX_ACTIVE` | `1` | Max concurrent active LLM inferences |
| `INFERENCE_MAX_QUEUE` | `3` | Max requests queued before rejecting |
| `INFERENCE_QUEUE_TIMEOUT`| `30.0` | Max wait time (seconds) in admission queue |
| `RATE_LIMIT_ANSWER_CAPACITY` | `2` | Burst capacity for `/answer` endpoints |
| `RATE_LIMIT_ANSWER_REFILL_SECONDS` | `30.0` | Token refill period for `/answer` |

---

## Contributing & Development Workflow

Contributions are welcomed. Please follow this standard engineering workflow:

1. **Fork the Repository**: Create an isolated fork under your GitHub account.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Run Code Quality & Tests**:
   Ensure all 62 tests pass and frontend builds cleanly before committing:
   ```bash
   python run_tests.py
   npm --prefix frontend run build
   ```
4. **Adhere to Code Standards**:
   - Python: Follow PEP 8 and use type annotations throughout.
   - TypeScript: Maintain strict type safety; avoid `any`.
   - Documentation: Maintain docstrings and update README when modifying contracts.
5. **Open a Pull Request**: Submit your PR targeting `main` with a detailed explanation of changes and test evidence.

For planned improvements and backlog discussions, refer to [docs/GITHUB_ISSUES.md](docs/GITHUB_ISSUES.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <b>Nexia</b> — Designed & Maintained by <a href="https://github.com/Deekshith1983">Deekshith1983</a> (<a href="mailto:dikideeku@gmail.com">dikideeku@gmail.com</a>)
</p>
