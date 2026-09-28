# Architectural Decision Records (ADRs)

> Log of key architectural decisions made in Nexia.

---

## ADR-001: Local-First Model Strategy with Gemma 3 4B via Ollama
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: The assistant must operate locally without mandatory external API keys or cloud privacy concerns.
- **Decision**: Standardize on Ollama running Google's `gemma3:4b` model. Its lightweight footprint allows efficient execution on consumer GPUs/CPUs with strong instruction-following capabilities.
- **Consequences**: Prompt design must be concise, structured, and resilient to small model reasoning limitations.

---

## ADR-002: Modular Pipeline Orchestrator (`NexiaCore`)
- **Date**: 2026-09-08
- **Status**: Accepted
- **Context**: Tightly coupling FastAPI route handlers to LLM calls makes testing and reasoning impossible.
- **Decision**: Encapsulate all pipeline operations inside a dedicated `NexiaCore` orchestrator class with individual single-responsibility modules (`QueryAnalyzer`, `ContextManager`, `GroundedGenerator`, `AnswerValidator`, `SourceAttributor`, `DecisionPolicy`).
- **Consequences**: Easy unit testing, modular maintenance, and transparent latency tracing.

---

## ADR-003: Deterministic Query Pre-Classification
- **Date**: 2026-09-12
- **Status**: Accepted
- **Context**: Calling the LLM to classify trivial greetings ("hi", "hello") and standard queries ("What is photosynthesis?") caused 2–3s unnecessary latency and occasional misclassifications.
- **Decision**: Implement a deterministic regex pre-classifier with 14 factual regex patterns and conversational checks in `QueryAnalyzer`. If regex matches, skip the LLM classification step completely (0ms latency).
- **Consequences**: Instant chit-chat responses and rock-solid factual query routing.

---

## ADR-004: Sentence-Level Evidence Verification
- **Date**: 2026-09-11
- **Status**: Accepted
- **Context**: LLMs can subtly hallucinate details even when provided with retrieved context.
- **Decision**: Pass generated answers through `AnswerValidator` to cross-reference each claim against the retrieved Wikipedia snippet corpus.
- **Consequences**: Guarantees verifiable answers; flags `INSUFFICIENT_EVIDENCE` or triggers graceful fallback when claims cannot be verified.

---

## ADR-005: Context & Pronoun Coreference Resolution
- **Date**: 2026-09-09
- **Status**: Accepted
- **Context**: Multi-turn questions like "When was he born?" need entity context from previous turns without mutating the user's displayed prompt.
- **Decision**: `ContextManager` resolves pronouns into an internal `resolved_query` used for retrieval and generation while storing the user's exact original text in the session history.
- **Consequences**: Preserves authentic conversation history while giving retrieval engines fully qualified queries.

---

*Last updated: 2026-09-22*
