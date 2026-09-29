# Nexia — Complete System Documentation

> **Nexia Grounded Conversational AI System**  
> **Production-Grade Architecture Specification (Stage 15 Complete)**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [The End-to-End Request Pipeline](#3-the-end-to-end-request-pipeline)
4. [Knowledge Strategy: Why No Vector Database?](#4-knowledge-strategy-why-no-vector-database)
5. [Subsystems & Component Breakdown](#5-subsystems--component-breakdown)
6. [Data Persistence & Database Architecture](#6-data-persistence--database-architecture)
7. [Configuration & Environment Management](#7-configuration--environment-management)
8. [API Reference & Endpoint Contracts](#8-api-reference--endpoint-contracts)
9. [Security, Guardrails & Isolation Guarantees](#9-security-guardrails--isolation-guarantees)
10. [Evaluation, Benchmarking & Robustness](#10-evaluation-benchmarking--robustness)
11. [Local Development & Operations Guide](#11-local-development--operations-guide)

---

## 1. Executive Summary

**Nexia** is a production-grade, grounded conversational AI assistant built to provide accurate, factual, and verifiable answers with **100% provenance and claim-level grounding**.

### Core Pillars
- **Strict Evidence Grounding**: The system does not fabricate facts or guess pretrained information. Every factual statement is backed by verifiable Wikipedia evidence.
- **Provider Modularity**: Model execution (Gemma 3 4B via Ollama) and external knowledge acquisition (Wikipedia) are abstracted behind explicit provider interfaces.
- **Adaptive Multi-Hop Reasoning**: Complex comparison and multi-entity queries are decomposed into structured information needs and navigated systematically.
- **Deterministic Fast-Paths**: Conversational greetings and exact entity queries bypass heavy LLM generation to achieve sub-millisecond response latency.
- **Production Resilience**: Transaction-safe SQLite storage (WAL mode), structured error handling, LRU caching, telemetry monitoring, and prompt injection immunity.

---

## 2. End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        React 19 Frontend (Vite)                        │
│   • Streaming Chat Interface   • Citation Badges   • Session Sidebar   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / Streaming Fetch
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend Engine                          │
│   • Lifespan Handlers   • Correlation ID (X-Request-Id) Middleware     │
│   • Global Error Sanitization   • Token Event Generator                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         Nexia Agent Runtime                            │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 1. Query Understanding (Deterministic Regex + LLM Classifier)  │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 2. Context & Coreference Manager (Resolves Pronouns)           │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 3. Capability Router (Direct Chat | Wikipedia Research | Memory│   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 4. Adaptive Decomposition Engine (Information Needs Graph)     │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 5. Wikipedia Navigator (Link Traversal, Cycle Prevention)      │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 6. KnowledgeWorkspace (Facts, Relationships, Conflict State)   │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 7. Grounded Generator (Evidence-Bounded Prompting via Gemma 3) │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 8. Claim Grounder & Answer Validator (Sentence Verification)   │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 9. Reflection & Answer Repair Engine (Targeted Re-Retrieval)   │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 10. Source Attribution & Citation Card Compilation             │   │
│   └────────────────────────────────────────────────────────────────┘   │
└───────────────────┬───────────────────────────────┬────────────────────┘
                    │                               │
                    ▼                               ▼
┌───────────────────────────────────┐ ┌──────────────────────────────────┐
│     Infrastructure Providers      │ │      SQLite Persistence Layer    │
│  • LanguageModelProvider (Ollama) │ │  • nexia_memory.db (Memories)    │
│  • KnowledgeProvider (Wikipedia)  │ │  • research_sessions.db (Tasks)  │
│  • LocalCacheManager (LRU Caches) │ │  • user_preferences.db (Profiles)│
│  • TelemetryTracker (Observability│ │  • evaluation_baselines.json     │
└───────────────────────────────────┘ └──────────────────────────────────┘
```

---

## 3. The End-to-End Request Pipeline

When a user prompt enters the system, it traverses the following 10-stage execution pipeline:

```text
User Message
    │
    ▼
[Stage 1: API & Correlation Middleware]
    │  - Attaches unique X-Request-Id (ContextVar).
    │  - Performs lightweight pre-flight cache health checks.
    ▼
[Stage 2: Query Understanding]
    │  - Tier 1: Deterministic regex fast-path (0ms) for greetings and standard queries.
    │  - Tier 2: LLM intent classifier (Extracts intent and target entities).
    ▼
[Stage 3: Context & Coreference Resolution]
    │  - Resolves pronouns ("he", "she", "it", "they") against conversation history.
    │  - Preserves user prompt while generating resolved retrieval query.
    ▼
[Stage 4: Capability Routing & Execution Planning]
    │  - Direct Chat Mode: Greetings bypass retrieval entirely.
    │  - Memory Mode: Explicit memory storage/retrieval commands.
    │  - Grounded Research Mode: Multi-stage Wikipedia investigation.
    ▼
[Stage 5: Adaptive Query Decomposition]
    │  - Deconstructs multi-entity and comparison questions into ordered InformationNeeds.
    │  - Establishes inter-need dependency chains.
    ▼
[Stage 6: Wikipedia Retrieval & Structured Navigation]
    │  - Exact title resolution, redirect normalization, disambiguation handling.
    │  - Article table of contents inspection, section extraction, and link navigation.
    │  - Cycle detection and depth budget governance.
    ▼
[Stage 7: KnowledgeWorkspace & Conflict Resolution]
    │  - Converts evidence into structured Subject-Predicate-Object facts.
    │  - Detects single-value contradictions (e.g. conflicting dates across sources).
    │  - Flags conflicts for epistemic hedging.
    ▼
[Stage 8: Grounded Generation]
    │  - Constructs an evidence-bounded prompt for Gemma 3 4B.
    │  - Applies user personalization preferences (concise, detailed, technical).
    │  - Deterministic fallback if evidence is absent ("I couldn't find that in the Wikipedia article.").
    ▼
[Stage 9: Claim-Level Validation & Answer Repair]
    │  - Decomposes generated answer into atomic claims.
    │  - Computes claim grounding status: SUPPORTED, UNSUPPORTED, PARTIAL, CONFLICTING.
    │  - Triggers ReflectionEngine if repair is needed (bounded retry budget).
    ▼
[Stage 10: Provenance Attribution & Streaming Response]
       - Assembles verified Wikipedia citations (URLs, titles, snippet previews).
       - Injects X-Nexia-Sources headers and streams tokens via FastAPI StreamingResponse.
```

---

## 4. Knowledge Strategy: Why No Vector Database?

Nexia deliberately avoids traditional vector databases and embedding-based RAG architectures:

| Traditional Vector RAG Problem | Nexia Architectural Solution |
| :--- | :--- |
| **Vector Drift & Semantic Mismatch**: Vector similarity frequently retrieves textually similar but factually irrelevant chunks. | **Deterministic API Search & Structure**: Uses exact title searches, redirect tables, and section-level Action APIs. |
| **Stale Embedding Indexes**: Pre-indexed vector databases become out of date and require expensive re-embedding pipelines. | **Live Knowledge Substrate**: Queries live Wikipedia directly with local in-memory LRU caching. |
| **Opaque Relevance Scores**: Float similarity thresholds (e.g. cosine distance `0.78`) provide no mathematical proof of truth. | **Structural KnowledgeWorkspace**: Converts text into Subject-Predicate-Object facts and verifies claims sentence-by-sentence. |
| **Contradiction Blindness**: Vector retrieval averages out conflicting facts without noticing contradictions. | **Explicit Conflict Detection**: Tracks single-value predicates (e.g., birth dates) and flags contradictory evidence for epistemic hedging. |

---

## 5. Subsystems & Component Breakdown

### 5.1 API Layer
- **`backend/main.py`**: FastAPI application entrypoint with `lifespan` initialization, CORS middleware, global exception handlers, and `/api/health`, `/api/live`, `/api/ready` monitoring probes.
- **`backend/app/routes/chat.py`**: Handles conversational streaming, client disconnect detection, and citation header generation.
- **`backend/app/routes/memory.py`**: REST API for inspecting, searching, updating, and soft-deleting persistent memories.

### 5.2 Nexia Core & Agent Runtime
- **`backend/app/core/nexia_core.py`**: Core pipeline orchestrator coordinating intent analysis, retrieval, generation, and verification.
- **`backend/app/core/query_understanding.py`**: Two-tier query classifier with 14 deterministic fast-path patterns.
- **`backend/app/core/context_manager.py`**: Multi-turn coreference and pronoun resolver.
- **`backend/app/core/decision_policy.py`**: Route selection (`direct_chat`, `grounded_knowledge`, `memory_interaction`).
- **`backend/app/agent/runtime.py`**: Runtime capability registry and execution coordinator.
- **`backend/app/agent/decomposition/`**: Query decomposition, dependency trees, and research plans.
- **`backend/app/agent/reflection/`**: Self-reflection engine evaluating validation gaps and executing repair cycles.

### 5.3 Knowledge, Grounding & Navigation
- **`backend/app/services/wikipedia_retriever.py`**: Wikipedia REST search, extract cleaner, and snippet ranker.
- **`backend/app/services/wikipedia_navigator.py`**: Article structure inspector, section retriever, internal link navigator, and cycle detector.
- **`backend/app/knowledge/workspace.py`**: In-memory knowledge graph containing entities, facts, relationships, and contradiction records.
- **`backend/app/knowledge/claims.py`**: Sentence-to-claim extractor and provenance mapper.
- **`backend/app/core/answer_validator.py`**: Fine-grained claim grounding and contradiction validator.
- **`backend/app/core/source_attribution.py`**: Citation compiler attaching URLs, section headers, and preview snippets.

### 5.4 Providers & Optimization
- **`backend/app/providers/model_provider.py`**: `LanguageModelProvider` abstraction wrapping Gemma 3 4B via Ollama.
- **`backend/app/providers/knowledge_provider.py`**: `KnowledgeProvider` abstraction wrapping Wikipedia retrieval and navigation.
- **`backend/app/optimization/cache.py`**: In-memory LRU caching for pages, model health, and validation results.
- **`backend/app/optimization/telemetry.py`**: Non-invasive operational telemetry tracking latency, call counts, and cache hit ratios.

---

## 6. Data Persistence & Database Architecture

Persistence is centralized in `backend/app/persistence/database.py` using SQLite with **Write-Ahead Logging (WAL)**, foreign key constraints, and **ACID transaction context managers**:

```python
with transaction(db_path) as conn:
    conn.execute(...)
    # Automatically commits on success, rolls back on exception
```

### Managed Databases

| Database File | Schema / Purpose | Key Features |
| :--- | :--- | :--- |
| **`data/nexia_memory.db`** | Structured long-term user memories and preferences. | Status lifecycles (`ACTIVE`, `SUPERSEDED`, `DELETED`), semantic keyword search, multi-tenant `user_id` filtering. |
| **`data/research_sessions.db`**| Long-horizon multi-hop research sessions. | Full JSON state snapshotting (information needs, evidence store, claims, navigation history), optimistic concurrency versioning. |
| **`data/user_preferences.db`** | User interaction profiles (response length, depth, formatting). | Rule-based preference resolution, explicit prompt overrides. |
| **`data/evaluation_baselines.json`** | Golden scenario metric baselines. | Local JSON storage for cross-milestone regression comparisons. |

---

## 7. Configuration & Environment Management

Centralized and validated in [`backend/app/config.py`](file:///c:/Users/dikid/OneDrive/Desktop/Nexia/backend/app/config.py) using Pydantic `BaseSettings`:

| Parameter | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| **Environment** | `APP_ENV` | `development` | `development`, `staging`, `production`, `test` |
| **Debug Mode** | `DEBUG` | `false` | Enables verbose pipeline tracing |
| **API Host / Port** | `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | FastAPI server binding |
| **Model Provider** | `MODEL_PROVIDER` | `ollama` | Active LLM provider (`ollama`, `mock`) |
| **Model Name** | `OLLAMA_MODEL` | `gemma3:4b` | Configured Ollama model identifier |
| **Ollama Base URL** | `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| **Model Timeout** | `OLLAMA_TIMEOUT` | `300.0` | Max generation timeout (seconds) |
| **Wikipedia Timeout** | `WIKIPEDIA_TIMEOUT` | `15.0` | Max Wikipedia HTTP timeout (seconds) |
| **Action Budget** | `RESEARCH_ACTION_BUDGET` | `8` | Maximum Wikipedia actions per query |
| **Navigation Depth** | `NAVIGATION_DEPTH` | `3` | Maximum internal link traversal hops |
| **Decomposition Limit**| `DECOMPOSITION_LIMIT` | `5` | Maximum sub-queries per complex request |
| **Repair Limit** | `REFLECTION_REPAIR_LIMIT` | `2` | Maximum answer repair retry iterations |
| **Cache Max Size** | `CACHE_MAX_SIZE` | `500` | In-memory LRU cache capacity |

---

## 8. API Reference & Endpoint Contracts

### 8.1 Chat Stream (`POST /api/chat`)
Executes the agentic pipeline and streams answer tokens.

- **Request Body**:
  ```json
  {
    "message": "Compare Albert Einstein and Isaac Newton's contributions to physics.",
    "conversation_id": "conv_abc123",
    "user_id": "user_default"
  }
  ```
- **Response**: `text/plain` chunked stream.
- **Response Headers**:
  - `X-Conversation-Id`: Active conversation session ID.
  - `X-Request-Id`: Unique correlation ID for tracing.
  - `X-Nexia-Sources`: JSON-encoded array of verified citation cards:
    ```json
    [
      {
        "title": "Albert Einstein",
        "url": "https://en.wikipedia.org/wiki/Albert_Einstein",
        "snippet": "Albert Einstein was a German-born theoretical physicist...",
        "section": "Physics contributions"
      }
    ]
    ```

### 8.2 Health & Monitoring
- **`GET /api/live`**: `{"status": "alive"}` (Liveness probe for process managers).
- **`GET /api/ready`**: Verifies database read/write and model readiness (Returns `200 OK` or `503 Service Unavailable`).
- **`GET /api/health`**: Comprehensive health status, version, uptime, and database metrics.

### 8.3 Long-Term Memory
- **`GET /api/memory?user_id=default`**: List active memories for a user.
- **`POST /api/memory`**: Store an explicit memory record.
- **`GET /api/memory/search?query=astrophysics&user_id=default`**: Semantic memory search.
- **`DELETE /api/memory/{memory_id}`**: Mark a memory as `DELETED`.

---

## 9. Security, Guardrails & Isolation Guarantees

1. **Multi-Tenant User Isolation**: Memory records, interaction profiles, and research sessions require strict `user_id` scoping in SQL queries. No cross-user data leakage.
2. **Untrusted Evidence Isolation**: Wikipedia content is treated as untrusted data. Retrieved text is strictly bounded inside data blocks in prompts; prompt injection attempts within Wikipedia text cannot override system instructions.
3. **No Arbitrary Code Execution**: User input and generated text are never passed to `eval()`, `exec()`, or sub-process shell commands.
4. **Path Traversal Protection**: Storage paths are validated against authorized project roots using `resolve_safe_path()`.
5. **No Private Chain-of-Thought Leaks**: Hidden model thoughts are never persisted or exposed over the API.

---

## 10. Evaluation, Benchmarking & Robustness

The evaluation subsystem lives in [`backend/app/evaluation/`](file:///c:/Users/dikid/OneDrive/Desktop/Nexia/backend/app/evaluation/) and remains strictly external to the production reasoning path:

- **`EvaluationHarness`**: Runs test scenarios against the agent runtime and collects metric telemetry.
- **14 Golden Benchmark Scenarios**: Covers simple factual queries, conversational fast-paths, exact entity lookups, contextual follow-ups, multi-entity comparisons, ungrounded rejection, conflict resolution, session resume, and personalization.
- **Deterministic Fault Injection**: Simulates Wikipedia network timeouts, Ollama offline states, malformed JSON responses, database locks, and client stream cancellations.
- **Baseline Metric Storage**: Compares latency, LLM call counts, Wikipedia requests, and cache hit ratios against saved baseline runs.

---

## 11. Local Development & Operations Guide

### Prerequisites
- Python 3.12+
- Node.js 20+
- Ollama with Gemma 3 4B installed (`ollama pull gemma3:4b`)

### Starting the System

1. **Start Ollama Engine**:
   ```bash
   ollama run gemma3:4b
   ```

2. **Start Backend API Server**:
   ```bash
   cd backend
   .venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

3. **Start React Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

4. **Verify System Status**:
   - Open browser at `http://localhost:5173`
   - Test Health API: `http://127.0.0.1:8000/api/health`
