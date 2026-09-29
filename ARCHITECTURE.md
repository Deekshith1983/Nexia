# Nexia Production Architecture — Canonical Reference

> **Nexia Grounded Agentic AI System**  
> **Final Architecture Specification (Stage 15 Complete)**

---

## 1. System Overview

Nexia is a production-grade, grounded, conversational AI system designed for rigorous, verifiable research and knowledge synthesis using Wikipedia as its sole external knowledge substrate and Gemma 3 4B via Ollama as its language model.

The architecture strictly enforces:
1. **Uncompromising Groundedness**: Every factual statement is backed by structural evidence with claim-level provenance.
2. **Provider Modularity**: Model and external knowledge providers are abstracted behind explicit interfaces.
3. **Multi-Horizon Reasoning**: Supports fast-path conversational queries, single-turn factual lookups, dynamic multi-hop research, intelligent Wikipedia navigation, and persistent research sessions.
4. **Resilience & Governance**: Centralized configuration, resource governance, deterministic failure injection, and graceful error containment.
5. **Strict Security Boundaries**: Full isolation between users, conversations, research sessions, and caches, with immunity to prompt injection and arbitrary execution.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application Layer                       │
│     • Lifespan Management  • Correlation ID  • Structured Errors       │
│     • Endpoints: /api/chat  /api/memory  /api/health  /api/ready       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Nexia Application Router                        │
│   • Request Validation  • Health Caching  • Stream Chunk Handling      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Nexia Agent Runtime Layer                          │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Capability Router: Direct Chat | Wikipedia Research | Memory  │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Multi-Hop Research Engine: Decomposition | Navigation | Action │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ KnowledgeWorkspace: Entities | Facts | Graph | Conflict State  │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Verification & Reflection: Claim Grounding | Repair Engine     │   │
│   └────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Infrastructure Provider Boundaries                   │
│   ┌─────────────────────────────────┐   ┌──────────────────────────┐   │
│   │      LanguageModelProvider      │   │    KnowledgeProvider     │   │
│   │   (Ollama / Gemma 3 4B Provider)│   │   (Wikipedia Provider)   │   │
│   └─────────────────────────────────┘   └──────────────────────────┘   │
│   ┌─────────────────────────────────┐   ┌──────────────────────────┐   │
│   │     Persistence Repositories    │   │  Optimization & Cache    │   │
│   │ (Memory, Sessions, Preferences) │   │ (LRU, Telemetry, Budget) │   │
│   └─────────────────────────────────┘   └──────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layered Subsystems & Responsibilities

### 2.1 API & Request Lifecycle (`backend/main.py`, `backend/app/routes/`)
- **Lifespan Management**: Centralized startup and shutdown handlers initializing logging, verifying persistence readiness, and clearing resources.
- **Correlation ID Middleware**: Propagates `X-Request-Id` across all async calls and logging records via `ContextVar`.
- **Global Error Handlers**: Maps structured domain errors (`NexiaError`, `ModelUnavailableError`, `PersistenceError`) to standard JSON responses without leaking internal file paths or stack traces.
- **Streaming Response**: True asynchronous token streaming with client disconnect detection (`raw_request.is_disconnected()`).

### 2.2 Nexia Core & Agent Runtime (`app/core/`, `app/agent/`)
- **ConversationManager**: Session-level turns, role tracking, and memory bounds.
- **QueryUnderstanding**: Two-tier query classification (deterministic regex fast-path + fallback model analyzer).
- **ContextManager**: Cross-turn coreference and entity resolution.
- **DecisionPolicy**: Route planning (`grounded_knowledge`, `direct_chat`, `memory_interaction`).
- **ResearchEngine & Decomposer**: Multi-hop planning, adaptive sub-query generation, and action budget governance.
- **NavigationEngine**: Intelligent link traversal, cycle detection, section extraction, and back-tracking prevention.
- **ReflectionEngine**: Verification of draft responses, detecting unsupported claims, triggering adaptive research or generation repair.

### 2.3 Knowledge & Grounding Architecture (`app/knowledge/`)
- **KnowledgeWorkspace**: Local knowledge graph tracking entities, relationships, predicate assertions, and conflict detection (single-value contradiction resolution).
- **Claim Grounding Engine**: Sentence-level claim extraction, mapping statements to supporting fact IDs, and computing grounding status (`SUPPORTED`, `UNSUPPORTED`, `PARTIALLY_SUPPORTED`, `CONFLICTING`).
- **SourceAttributor**: Immutable provenance attaching Wikipedia URLs, article titles, section anchors, and text snippets.

### 2.4 Persistent Memory & Personalization (`app/memory/`, `app/personalization/`, `app/sessions/`)
- **MemoryManager**: Long-term fact and preference persistence with semantic keyword search, status lifecycles (`ACTIVE`, `SUPERSEDED`, `DELETED`), and user isolation.
- **PersonalizationEngine**: Style and depth preferences dynamically resolved with explicit user overrides taking precedence.
- **ResearchSessionManager**: Long-horizon research tracking with transaction-safe SQLite persistence surviving server restarts.

### 2.5 Infrastructure Provider Boundaries (`app/providers/`)
- **`LanguageModelProvider`**: Clean abstraction defining `generate()`, `stream()`, and `health()`. Wraps Gemma 3 4B via Ollama.
- **`KnowledgeProvider`**: Clean abstraction defining `retrieve()`, `navigate_step()`, and `health()`. Wraps Wikipedia API services.

### 2.6 Persistence Layer (`app/persistence/`)
- **`database.py`**: Centralized SQLite connection management with WAL journal mode, busy timeouts, foreign key enforcement, and atomic transaction context managers (`with transaction(...)`).

### 2.7 Optimization & Resource Governance (`app/optimization/`)
- **LocalCacheManager**: In-memory LRU caching for Wikipedia page extracts, search results, model readiness, and validation outputs.
- **TelemetryTracker**: Non-invasive operational telemetry recording latency, token generation speed, cache hit ratios, and action counts.
- **Concurrency & Budget Limits**: Fixed budgets preventing unbounded research loops, recursion, or memory leaks.

### 2.8 Evaluation & Verification Layer (`app/evaluation/`)
- Dedicated out-of-band evaluation harness executing golden scenario benchmarks, grounding verifications, fault-injection tests, and regression baselines without polluting the runtime reasoning path.

---

## 3. Security & Isolation Guarantees

| Boundary | Enforcement Mechanism |
| :--- | :--- |
| **User Isolation** | All memory, preferences, and session queries require strict `user_id` filtering. No cross-user leakage. |
| **Untrusted Content** | Retrieved Wikipedia content is treated as untrusted data and strictly bounded inside evidence prompt blocks. |
| **Prompt Injection** | Input and evidence text containing instruction-like tokens cannot override system instructions or alter action budgets. |
| **Provenance Integrity** | Claims require verified fact IDs matching retrieved evidence IDs. Fabricated provenance is rejected. |
| **Code Execution** | Evaluated expressions and user inputs are strictly string-parsed; no `eval()`, `exec()`, or dynamic shell calls. |
| **Path Traversal** | Storage paths are resolved within authorized root directories via `resolve_safe_path()`. |

---

## 4. Local Execution & Verification

### Running the Production Server
```bash
cd backend
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Verification Endpoints
- **Liveness**: `GET http://127.0.0.1:8000/api/live`
- **Readiness**: `GET http://127.0.0.1:8000/api/ready`
- **Detailed Health**: `GET http://127.0.0.1:8000/api/health`
- **Chat**: `POST http://127.0.0.1:8000/api/chat`
- **Memory**: `GET / POST http://127.0.0.1:8000/api/memory`
