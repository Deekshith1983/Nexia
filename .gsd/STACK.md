# Technology Stack

> Nexia Technology Stack & Inventory
> Canonical Reference for GSD

---

## Runtime

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | Backend language |
| Web Framework | FastAPI | >=0.110.0 | Async REST API server |
| ASGI Server | Uvicorn | >=0.28.0 | High-performance ASGI web server |
| Frontend Runtime | Node.js | 18+ | Frontend build & tooling environment |
| UI Framework | React | 19.2.8 | Declarative user interface |
| Build Tool | Vite | 8.2.2 | Fast frontend bundler and dev server |
| Local LLM Engine | Ollama | >=0.1.8 | Local model runner hosting `gemma3:4b` |

---

## Core Technologies & Libraries

### Backend
| Package | Version | Purpose |
|---|---|---|
| `fastapi` | >=0.110.0 | Web API routing & middleware |
| `uvicorn` | >=0.28.0 | ASGI web server |
| `httpx` | >=0.27.0 | Async HTTP client for Wikipedia REST API and Ollama |
| `pydantic` | >=2.6.0 | Data validation, schemas, and settings management |
| `python-dotenv` | >=1.0.1 | Environment configuration |
| `ollama` | >=0.1.8 | Official Python bindings for Ollama |
| `langchain-core` / `langchain-ollama` | >=0.0.1 | Extended LLM interfacing utilities |

### Frontend
| Package | Version | Purpose |
|---|---|---|
| `react` / `react-dom` | ^19.2.8 | Component rendering & state management |
| `lucide-react` | ^1.41.0 | Modern SVG icon set |
| `react-markdown` | ^10.1.0 | Markdown parsing & safe HTML rendering |
| `remark-gfm` | ^4.0.1 | GitHub-flavored markdown support (tables, lists, stikethroughs) |
| `vite` | ^8.2.2 | Development server and production bundling |
| `oxlint` | ^1.79.0 | Ultra-fast JS/TS linter |

---

## File Inventory

| Category | Directory | Key Files | Purpose |
|---|---|---|---|
| Backend Core | `backend/app/core/` | 9 files | Orchestrator, analyzer, context, generator, validator, attribution |
| Backend Services | `backend/app/services/` | 2 files | Wikipedia retriever, Ollama client service |
| Backend Routes | `backend/app/routes/` | 1 file | Chat endpoints (`chat.py`) |
| Frontend UI | `frontend/src/` | 4 files | App component, styling, entrypoints |
| Workflows & Skills | `.agent/`, `.agents/` | 27 workflows, 14 skills | GSD automation & subagent protocols |

---

*Last updated: 2026-09-22*
