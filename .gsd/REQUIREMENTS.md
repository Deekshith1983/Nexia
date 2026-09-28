# REQUIREMENTS.md — System Requirements

> Traceable requirements for Nexia.

---

## Functional Requirements

| ID | Requirement | Category | Status | Verification |
|---|---|---|---|---|
| REQ-01 | Support multi-turn chat sessions with distinct session IDs | Foundation | ✅ Completed | Automated / Manual UI test |
| REQ-02 | Deterministic classification for casual greetings without LLM overhead | Intelligence | ✅ Completed | 0ms Qu trace, test suite |
| REQ-03 | Extract key entities from factual questions | Intelligence | ✅ Completed | Unit test regex & LLM parser |
| REQ-04 | Resolve pronoun references (`he`, `she`, `it`, `they`) to previous turn entities | Context | ✅ Completed | Multi-turn test suite |
| REQ-05 | Retrieve relevant summaries and extracts from Wikipedia via REST API | Retrieval | ✅ Completed | HTTP status 200, extract parse |
| REQ-06 | Generate natural-language answers strictly bound by retrieved evidence | Generation | ✅ Completed | Prompt inspection & test suite |
| REQ-07 | Validate generated claims against retrieved evidence snippets | Validation | ✅ Completed | Validator score & status |
| REQ-08 | Return source metadata (URL, title, snippet) with response | Attribution | ✅ Completed | Source card rendering in UI |
| REQ-09 | Log sub-millisecond per-stage execution traces for every request | Observability | ✅ Completed | `[NEXIA TRACE]` console output |
| REQ-10 | Pluggable multi-source retriever aggregation (Web, ArXiv, Docs) | Future (Phase 3) | ⬜ Pending | Phase 3 plan |
| REQ-11 | Local tool/function execution (Calculator, Python sandbox) | Future (Phase 4) | ⬜ Pending | Phase 4 plan |
| REQ-12 | Persistent SQLite conversation and memory storage | Future (Phase 5) | ⬜ Pending | Phase 5 plan |

---

*Last updated: 2026-09-22*
