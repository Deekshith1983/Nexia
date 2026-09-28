# SPEC.md — Project Specification

> **Status**: `FINALIZED`
> **Project**: Nexia — Local Agentic Factual AI Assistant
> **Milestone**: v0.2.0 (Milestone 2 Complete)

---

## Vision

Nexia is a high-performance, private, local-first agentic conversational AI assistant designed to deliver factual, verifiable, and grounded responses without hallucinations. Powered by local LLMs (Gemma 3 4B via Ollama) paired with deterministic retrieval, coreference resolution, sentence-level evidence verification, and transparent source attribution, Nexia combines low-latency conversational fluency with research-grade factual accuracy.

---

## Goals

1. **Local-First Privacy & Autonomy** — Execute all core intelligence, retrieval processing, inference, and session management locally without mandatory cloud dependencies.
2. **Grounded & Verifiable Generation** — Ensure factual queries are answered strictly using retrieved evidence (e.g. Wikipedia knowledge) with automated claim validation and source citation cards.
3. **Multi-Turn Context & Coreference Understanding** — Seamlessly resolve contextual references (e.g. pronouns like "he", "she", "it", "they") to previous conversation entities while preserving original user queries.
4. **Deterministic & Fast Orchestration** — Leverage deterministic pre-classification for common knowledge queries and casual chit-chat to minimize LLM overhead and achieve sub-second response times.
5. **Modern, Responsive User Interface** — Provide an intuitive, glassmorphic, and accessible chat UI built in React + Vite with live timing traces, source cards, and session persistence.

---

## Non-Goals (Out of Scope for Milestone 2)

- Cloud multi-tenant authentication / SaaS billing.
- Arbitrary untrusted Python code execution sandbox (slated for Milestone 4).
- Autonomous web browser automation (slated for future milestones).
- Multi-modal image/voice processing (slated for Milestone 6).

---

## Constraints

- **Hardware / Inference**: Optimized for consumer hardware running local Ollama instances (primary target: `gemma3:4b`).
- **Backend**: Python 3.10+ with FastAPI and asynchronous I/O.
- **Frontend**: React 19, Vite, Vanilla CSS design system.
- **Verification Strictness**: Zero tolerance for unverified claims in `grounded_knowledge` mode; must flag `INSUFFICIENT_EVIDENCE` or provide fallback if evidence is lacking.

---

## Success Criteria

- [x] Multi-turn chat interface with real-time streaming and session switching.
- [x] Deterministic intent classification for casual greetings and common query patterns (0ms LLM overhead).
- [x] Multi-turn entity tracking and pronoun resolution ("Who is Einstein?" -> "When was he born?").
- [x] Direct Wikipedia REST API retrieval with snippet cleaning and evidence ranking.
- [x] Strict evidence-grounded prompt generation with Gemma 3 4B.
- [x] Sentence-level answer validation against retrieved Wikipedia snippets.
- [x] Structured source attribution cards with direct article URLs.
- [x] Comprehensive test suite with 100% pass rate across coreference, retrieval, generation, validation, and chit-chat.

---

*Last updated: 2026-09-22*
