---
milestone: Milestone 2 Complete (Grounded Knowledge Core)
version: 0.2.0
updated: 2026-09-22T17:10:00+05:30
---

# ROADMAP.md

> **Current Phase:** Phase 2 Complete (Milestone 2 Verified)
> **Next Phase:** Phase 3 — Multi-Source Retrieval & Extended Reasoning
> **Status:** ✅ Milestone 2 Complete

---

## Must-Haves (from SPEC)

- [x] Multi-turn conversational session management (`ConversationManager`)
- [x] Fast deterministic query classification & intent routing (`QueryAnalyzer`)
- [x] Coreference resolution for contextual pronouns (`ContextManager`)
- [x] Direct Wikipedia REST API retrieval & snippet ranking (`WikipediaRetriever`)
- [x] Evidence-grounded answer generation via Gemma 3 4B (`GroundedGenerator`)
- [x] Sentence-level hallucination verification (`AnswerValidator`)
- [x] Clean source provenance & citation cards (`SourceAttributor`)
- [x] Unified decision policy & sub-millisecond trace logging (`DecisionPolicy`, `NexiaCore`)
- [x] React 19 glassmorphic user interface with live source cards and timing metrics

---

## Completed Phases

### Phase 1: Foundation & Local Stack Setup
**Status:** ✅ Complete  
**Objective:** Establish core architecture with React + Vite frontend, FastAPI backend, and local Ollama + Gemma 3 4B inference.  
**Deliverables:**
- [x] FastAPI async backend with `/api/chat`, `/api/health`, and model probing
- [x] React 19 SPA with chat bubbles, session sidebar, and streaming support
- [x] Ollama client connection with fallback checks

---

### Phase 2: Nexia Core Architecture (Grounded Factual Intelligence)
**Status:** ✅ Complete  
**Objective:** Build deterministic, multi-stage retrieval-augmented conversational pipeline with coreference resolution and evidence verification.  
**Delivered Stages:**
- [x] **Stage 2.1 — Nexia Core Foundation**: Central orchestrator pipeline (`NexiaCore`) and modular interfaces.
- [x] **Stage 2.2 — Conversation State**: Multi-turn history, session isolation, context pruning.
- [x] **Stage 2.3 — Query Understanding**: Intent classification (`factual`, `conversational`, `ambiguous`), entity extraction, search query reformulation.
- [x] **Stage 2.4 — Context & Coreference Resolution**: Resolving contextual pronouns (`he`, `she`, `it`, `they`) without altering original user queries.
- [x] **Stage 2.5 — Wikipedia Retrieval**: REST-based Wikipedia fetcher with section parsing, lead extraction, and fallback search.
- [x] **Stage 2.6 — Grounded Generation**: Strict evidence-constrained prompt templating with Gemma 3 4B.
- [x] **Stage 2.7 — Answer Validation**: Sentence-level claim verification against retrieved evidence snippets.
- [x] **Stage 2.8 — Source Attribution & Citations**: Direct Wikipedia page metadata, article title, and link attachment.
- [x] **Stage 2.9 — Response Orchestration & Decision Policy**: Coherent route execution (`grounded_knowledge`, `direct_chat`, `fallback`) with sub-millisecond latency traces.
- [x] **Stabilization Pass**: Deterministic regex pre-classifier (0ms for chit-chat and standard questions), relaxed evidence validation parser, and end-to-end 16/16 test suite.

---

## Upcoming Phases

### Phase 3: Multi-Source Retrieval & Multi-Hop Reasoning
**Status:** ⬜ Not Started  
**Objective:** Broaden retrieval capabilities beyond Wikipedia to include local documents, web search (DuckDuckGo/Brave API), and scientific paper summaries (ArXiv), with multi-hop query decomposition.  
**Planned Stages:**
- [ ] Stage 3.1: Pluggable retriever interface & multi-source aggregator
- [ ] Stage 3.2: Web search integration (DuckDuckGo / Brave Search API)
- [ ] Stage 3.3: Local PDF / Markdown document ingestion & local vector search
- [ ] Stage 3.4: Multi-hop query decomposition and synthesized multi-source evidence

---

### Phase 4: Agentic Tool Use & Action Execution
**Status:** ⬜ Not Started  
**Objective:** Enable Nexia to use tools dynamically (calculator, Python code runner, date/time tools, file system operations) with human confirmation checkpoints.  
**Planned Stages:**
- [ ] Stage 4.1: Deterministic function/tool schema definition
- [ ] Stage 4.2: Local Python code execution sandbox
- [ ] Stage 4.3: Tool call orchestrator with verification & safety checkpoints

---

### Phase 5: Persistent Memory & Knowledge Graph
**Status:** ⬜ Not Started  
**Objective:** Persistent cross-session user memory, personal knowledge graph, and user preference retention with local SQLite / ChromaDB.  
**Planned Stages:**
- [ ] Stage 5.1: SQLite persistent conversation database
- [ ] Stage 5.2: Entity relationship extraction and local knowledge graph
- [ ] Stage 5.3: Long-term memory recall and user profile synthesis

---

### Phase 6: Multi-Modal Interaction & Voice Integration
**Status:** ⬜ Not Started  
**Objective:** Real-time local voice input/output (Whisper.cpp + Piper TTS) and image understanding.  
**Planned Stages:**
- [ ] Stage 6.1: Local Whisper speech-to-text integration
- [ ] Stage 6.2: Local Piper TTS audio generation
- [ ] Stage 6.3: Vision model integration for document/image queries

---

## Progress Summary

| Phase | Milestone | Status | Stages Complete |
|---|---|---|---|
| Phase 1: Foundation Setup | v0.1.0 | ✅ Complete | 3/3 |
| Phase 2: Nexia Core (Grounded Knowledge) | v0.2.0 | ✅ Complete | 10/10 |
| Phase 3: Multi-Source Retrieval | v0.3.0 | ⬜ Not Started | 0/4 |
| Phase 4: Agentic Tool Use | v0.4.0 | ⬜ Not Started | 0/3 |
| Phase 5: Persistent Memory & Graph | v0.5.0 | ⬜ Not Started | 0/3 |
| Phase 6: Multi-Modal & Voice | v0.6.0 | ⬜ Not Started | 0/3 |

---

*Canonical file: `.gsd/ROADMAP.md` | Last updated: 2026-09-22*
