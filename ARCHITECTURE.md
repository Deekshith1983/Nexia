# Architecture

> Nexia System Architecture — Milestone 2
> Canonical Reference for GSD

---

## Overview

Nexia is structured as a layered agentic system where the frontend presents a responsive chat workspace, the FastAPI backend exposes asynchronous endpoints, and `NexiaCore` orchestrates modular intelligence components for deterministic routing, retrieval, generation, verification, and citation.

```
┌───────────────────────────────────────────────────────────────┐
│                    React 19 Frontend (Vite)                   │
│   • Session Sidebar  • Message Stream  • Citation Cards      │
│   • Latency & Validation Badges  • Glassmorphic Dark UI      │
└───────────────────────────────┬───────────────────────────────┘
                                │ HTTP / JSON API
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                     FastAPI Async Router                      │
│            /api/chat  /api/health  /api/sessions              │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                          Nexia Core                           │
│   ┌───────────────────────────────────────────────────────┐   │
│   │                 ConversationManager                   │   │
│   │   • Session State  • History Buffer  • Turn Counter   │   │
│   └───────────────────────────┬───────────────────────────┘   │
│                               ▼                               │
│   ┌───────────────────────────────────────────────────────┐   │
│   │                   QueryAnalyzer                       │   │
│   │   • Deterministic Pre-Classifier (0ms Regex Fast-Path)│   │
│   │   • LLM Fallback Classifier (Intent + Entity extract) │   │
│   └───────────────────────────┬───────────────────────────┘   │
│                               ▼                               │
│   ┌───────────────────────────────────────────────────────┐   │
│   │                   ContextManager                      │   │
│   │   • Coreference Resolution (he/she/it -> Entity)      │   │
│   │   • Resolved Query Formulation (keeps original query) │   │
│   └───────────────────────────┬───────────────────────────┘   │
│                               ▼                               │
│   ┌───────────────────────────────────────────────────────┐   │
│   │                   DecisionPolicy                      │   │
│   │   • Routing: direct_chat vs grounded_knowledge        │   │
│   └──────┬────────────────────────────────────────┬───────┘   │
│          │ (direct_chat)                          │ (grounded)│
│          ▼                                        ▼           │
│   ┌───────────────┐               ┌───────────────────────┐   │
│   │  LLMInterface │               │   WikipediaRetriever  │   │
│   │  Direct Chat  │               │   REST Search/Extract │   │
│   └──────┬────────┘               └───────────┬───────────┘   │
│          │                                    ▼               │
│          │                        ┌───────────────────────┐   │
│          │                        │   GroundedGenerator   │   │
│          │                        │   Evidence Prompting  │   │
│          │                        └───────────┬───────────┘   │
│          │                                    ▼               │
│          │                        ┌───────────────────────┐   │
│          │                        │    AnswerValidator    │   │
│          │                        │    Claim Verification │   │
│          │                        └───────────┬───────────┘   │
│          │                                    ▼               │
│          │                        ┌───────────────────────┐   │
│          │                        │   SourceAttributor    │   │
│          │                        │   Article Links/Cards │   │
│          │                        └───────────┬───────────┘   │
│          │                                    │               │
│          └────────────────┬───────────────────┘               │
│                           ▼                                   │
│              Structured Response + Trace Logs                 │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│               Local Ollama Engine (Gemma 3 4B)                │
└───────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. NexiaCore Orchestrator
- **Location:** `backend/app/core/nexia_core.py`
- **Purpose:** Central coordinator executing request lifecycle, timing instrumentation, error containment, and trace generation.
- **Trace Output:** Logs `[NEXIA TRACE]` with per-stage latencies (`qu`, `ret`, `gen`, `val`, `total`).

### 2. ConversationManager
- **Location:** `backend/app/core/conversation_manager.py`
- **Purpose:** Tracks in-memory conversation sessions, message histories, role attribution (`user`, `assistant`, `system`), and context windows.

### 3. QueryAnalyzer
- **Location:** `backend/app/core/query_understanding.py`
- **Purpose:** Analyzes query intent (`factual`, `conversational`, `ambiguous`), extracts target entities, and reformulates retrieval queries.
- **Pattern:** Two-tier classification (deterministic fast-path with 14 regex patterns for instant zero-LLM classification, plus LLM fallback).

### 4. ContextManager
- **Location:** `backend/app/core/context_manager.py`
- **Purpose:** Resolves cross-turn coreference pronouns (e.g. "When was he born?" following "Who is Albert Einstein?") into fully resolved queries while maintaining clean user message history.

### 5. WikipediaRetriever
- **Location:** `backend/app/services/wikipedia_retriever.py`
- **Purpose:** Performs REST API searches against Wikipedia, cleans raw HTML/wikitext extracts, splits into evidence sections, and ranks relevant passages.

### 6. GroundedGenerator
- **Location:** `backend/app/core/grounded_generator.py`
- **Purpose:** Generates natural language answers strictly bound by provided evidence context using Gemma 3 4B, rejecting external speculation.

### 7. AnswerValidator
- **Location:** `backend/app/core/answer_validator.py`
- **Purpose:** Performs sentence-by-sentence verification of generated answers against retrieved evidence, returning validation status (`VALIDATED`, `INSUFFICIENT_EVIDENCE`, `VALIDATION_ERROR`).

### 8. SourceAttributor
- **Location:** `backend/app/core/source_attribution.py`
- **Purpose:** Attaches immutable provenance metadata (Wikipedia URL, article title, snippet preview) directly to the response payload.

### 9. DecisionPolicy
- **Location:** `backend/app/core/decision_policy.py`
- **Purpose:** Determines execution route (`grounded_knowledge`, `direct_chat`, `fallback`) and controls whether retrieval or validation is required.

---

*Canonical file: `.gsd/ARCHITECTURE.md` | Last updated: 2026-09-22*
