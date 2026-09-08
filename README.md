# Nexia — Grounded Wikipedia AI Agent

Nexia is a conversational AI agent designed around its own reasoning and retrieval architecture (Nexia Core), using Wikipedia as its initial external knowledge source and a local language model (Gemma 3 4B) for language generation.

---

## 📊 Current Development Status

**Nexia V1** is currently in **Stage 2 — LLM Integration, Development & Testing**.

| Stage | Description | Status |
|---|---|---|
| **Stage 1** | Environment & Dependencies | ✅ Completed |
| **Stage 3** | FastAPI + React Chat UI Scaffold | ✅ Completed |
| **Stage 2** | LLM Integration, Streaming & Nexia Core Separation | 🔄 In Progress |

---

## 🏗 Architecture Overview

Nexia is **not** simply a UI wrapper around Gemma, nor is it a simple prompt template. It is an independent AI system designed around a language model component.

### Target Nexia Core Architecture (Long-term)

```text
React UI
    ↓
FastAPI
    ↓
Nexia Core
    ├── Conversation Manager
    ├── Query Understanding
    ├── Context/Coreference Manager
    ├── Wikipedia Retriever
    ├── Knowledge Processing
    ├── Reasoning/Decision Engine
    ├── Answer Validation
    └── Language Generation
            ↓
        Gemma 3 4B
```

### Stage 2 Architectural Objective (Near-term)

Establishing a clean separation between **Nexia application logic** and **LLM infrastructure**:

```text
React UI → FastAPI → Nexia Core → Language Model Service → Gemma 3 4B
```

---

## 🔒 Constraints & Principles

1. **LLM Runtime**: Gemma 3 4B (`gemma3:4b`) via local [Ollama](https://ollama.com).
2. **Knowledge Source**: Wikipedia ONLY during the initial grounded phase.
3. **Grounding Principle**: Answers are based on retrieved evidence without fabricating facts absent from context.
4. **Stage-by-Stage Methodology**: Development progresses strictly by approved stages.

---

## 🚀 Quick Start Guide

### 1. Backend Setup (FastAPI)

```bash
cd backend
# Activate Python 3.12 virtual environment
.\.venv\Scripts\activate

# Run FastAPI server
python main.py
```
Backend API runs on `http://127.0.0.1:8000`.

### 2. Frontend Setup (React + Vite)

```bash
cd frontend
npm run dev
```
React Chat UI runs on `http://localhost:5173`.

### 3. Local LLM Runtime (Ollama + Gemma 3 4B)

1. Download and run [Ollama](https://ollama.com).
2. Pull and run the model:
   ```bash
   ollama run gemma3:4b
   ```
3. Nexia backend connects to Ollama at `http://localhost:11434`.
