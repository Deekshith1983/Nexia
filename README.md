# Nexia — Wikipedia-Grounded Conversational AI Assistant

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3+-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5+-3178c6.svg)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.3+-green.svg)](https://github.com/pgvector/pgvector)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-orange.svg)](https://ollama.com/)
[![Gemma](https://img.shields.io/badge/LLM-Gemma_3_4B-purple.svg)](https://ollama.com/library/gemma3:4b)

Nexia is a Wikipedia-grounded conversational AI application. It combines a React and TypeScript chat interface with a FastAPI backend that retrieves information from Wikipedia, stores processed content and embeddings in PostgreSQL with pgvector, and uses locally hosted models through Ollama to generate evidence-grounded answers.

The project is designed around a separation between **retrieval**, **evidence preparation**, and **answer generation**. The language model synthesizes an answer from retrieved context; Wikipedia remains the configured knowledge source for the documented workflow.

> **Knowledge-source scope:** This README describes the Wikipedia-based knowledge workflow represented by this project. It does not imply that Nexia has general web search or access to arbitrary external knowledge sources.

---

## Contents

1. [Overview](#1-overview)
2. [Features](#2-features)
3. [Technology stack](#3-technology-stack)
4. [Architecture](#4-architecture)
5. [How a question is processed](#5-how-a-question-is-processed)
6. [Repository layout](#6-repository-layout)
7. [Prerequisites](#7-prerequisites)
8. [Installation and setup](#8-installation-and-setup)
9. [Run Nexia](#9-run-nexia)
10. [Testing](#10-testing)
11. [API overview](#11-api-overview)
12. [Retrieval configuration](#12-retrieval-configuration)
13. [Engineering decisions](#13-engineering-decisions)
14. [Known limitations and future work](#14-known-limitations-and-future-work)
15. [Contributing](#15-contributing)
16. [Development workflow](#16-development-workflow)
17. [License](#17-license)
18. [Author and attribution](#18-author-and-attribution)

---

## 1. Overview

Nexia provides a conversational interface for asking questions against a Wikipedia-based knowledge collection. Instead of relying only on the language model's pretrained knowledge, the backend retrieves relevant article content, selects passages, and supplies that evidence as context for answer generation.

The application has three main responsibilities:

- **Frontend:** provides the chat experience, conversation navigation, message rendering, and streaming response presentation.
- **Backend:** interprets requests, retrieves and prepares evidence, coordinates local model inference, validates/formats responses, and manages conversation operations.
- **Data and model services:** PostgreSQL stores application data and retrieved content; pgvector supports vector similarity retrieval; Ollama serves the configured language and embedding models.

Nexia is intended to run locally. The documented model integration uses Ollama with `gemma3:4b` for answer generation and `nomic-embed-text` for embeddings.

## 2. Features

The following describes the capabilities represented in the supplied project documentation. Confirm the current implementation and configuration when changing or extending these features.

### Conversational interface

- GPT-style chat interface built with React and TypeScript.
- Responsive layout with conversation navigation.
- Conversation creation, listing, retrieval, renaming, and deletion through backend APIs.
- Progressive answer delivery through Server-Sent Events (SSE), where the streaming endpoint is used.
- Markdown-based answer presentation, including formatted text and links.

### Wikipedia retrieval and evidence preparation

- Wikipedia article discovery and retrieval.
- Article content processing and passage chunking.
- Persistent document and passage storage in PostgreSQL.
- Text embeddings generated through Ollama.
- Lexical retrieval using BM25.
- Semantic retrieval using pgvector.
- Hybrid ranking that combines lexical and vector retrieval.
- Evidence selection and packing before answer generation.

### Grounded answer generation

- Local answer generation through Ollama.
- Evidence-aware prompt construction.
- Answer verification using the existing grounding/claim-validation components.
- A single relevant Wikipedia article hyperlink appended at the end of a completed answer, when valid source metadata is available.

### Operational capabilities

- Centralized retrieval configuration for question-answering requests.
- Normal and streaming answer endpoints.
- Conversation persistence.
- Health endpoint for checking backend availability.

---

## 3. Technology stack

The versions below reflect the versions or minimum versions stated in the supplied project documentation. The dependency manifests and lockfiles should be treated as authoritative if they differ.

### Frontend

| Area | Technology | Responsibility |
|---|---|---|
| UI framework | React 18 | Component-based user interface |
| Language | TypeScript 5.5+ | Typed frontend code and API contracts |
| Build tool | Vite 6 | Development server and production build |
| Styling | Tailwind CSS 3.4+ | Utility-based styling |
| UI state | Zustand 5 | Client-side interface state |
| Server state | TanStack Query 5 | API data fetching and cache synchronization |
| Icons | Lucide React | Interface icons |
| Markdown | React Markdown and Rehype | Rendering generated Markdown |
| Frontend tests | Vitest and React Testing Library | Component and frontend behavior tests |

### Backend

| Area | Technology | Responsibility |
|---|---|---|
| Language | Python 3.12+ | Backend implementation |
| API framework | FastAPI | HTTP API and streaming responses |
| Validation | Pydantic v2 | Request and response validation |
| ORM | SQLAlchemy 2 | Database models and database access |
| PostgreSQL driver | asyncpg | Asynchronous PostgreSQL connectivity |
| NLP | spaCy | Linguistic analysis and entity extraction |
| Backend tests | pytest and pytest-asyncio | Backend unit and integration tests |

### Database and AI

| Area | Technology | Responsibility |
|---|---|---|
| Relational database | PostgreSQL 15+ | Conversations, messages, documents, and related records |
| Vector extension | pgvector | Storage and similarity search for embeddings |
| Model runtime | Ollama | Local model serving |
| Answer model | `gemma3:4b` | Answer generation |
| Embedding model | `nomic-embed-text` | Text and query embeddings |
| Lexical retrieval | BM25 | Keyword-based passage retrieval |
| Hybrid ranking | BM25 + vector retrieval | Combines lexical and semantic relevance |

---

## 4. Architecture

Nexia uses a frontend/backend architecture. The frontend sends user questions to the FastAPI backend. The backend prepares evidence from the configured Wikipedia knowledge source, invokes the local model through Ollama, and returns a response to the frontend.

### Request and response sequence

```mermaid
sequenceDiagram
    actor User
    participant UI as React frontend
    participant API as FastAPI
    participant Store as PostgreSQL / pgvector
    participant Wiki as Wikipedia
    participant Ollama as Ollama
    participant Verify as Validation and formatting

    User->>UI: Submit a question
    UI->>API: Send answer request
    API->>API: Validate retrieval configuration
    API->>Store: Look up available article data

    alt Required article data is missing
        API->>Wiki: Retrieve article content
        Wiki-->>API: Article content
        API->>API: Process and chunk content
        API->>Ollama: Generate embeddings
        Ollama-->>API: Embeddings
        API->>Store: Persist documents, chunks, and embeddings
    end

    API->>Store: Retrieve lexical and vector candidates
    Store-->>API: Candidate passages
    API->>API: Rank and prepare evidence
    API->>Ollama: Generate answer using evidence
    Ollama-->>API: Generated answer
    API->>Verify: Validate and format answer
    Verify-->>API: Clean answer and final article link
    API->>Store: Persist conversation turn
    API-->>UI: Return answer or stream response
    UI-->>User: Display answer and final citation
```

The diagrams describe the intended documented flow. Exact internal sequencing can vary with cache hits, streaming behavior, and the implementation of individual services.

---

### Main architectural responsibilities

**Frontend**
- Collects user questions.
- Manages chat and conversation interactions.
- Calls the backend API.
- Displays normal and streamed responses.
- Renders Markdown and the final article hyperlink.

**FastAPI backend**
- Exposes health, conversation, and answer-generation endpoints.
- Validates incoming requests.
- Coordinates retrieval, evidence preparation, inference, and response formatting.
- Handles streaming and cancellation routes where implemented.

**Retrieval layer**
- Analyzes the question.
- Identifies relevant entities or sub-questions.
- Retrieves candidate passages using lexical and vector search.
- Ranks, filters, and packs evidence for the language model.

**PostgreSQL and pgvector**
- Persist conversations and messages.
- Store retrieved article content and passage data.
- Store embeddings for semantic retrieval.

**Ollama**
- Provides local model inference.
- Serves the configured answer-generation and embedding models.

### Request and response sequence

```mermaid
sequenceDiagram
    actor User
    participant UI as React frontend
    participant API as FastAPI
    participant Store as PostgreSQL / pgvector
    participant Wiki as Wikipedia
    participant Ollama as Ollama
    participant Verify as Validation and formatting

    User->>UI: Submit a question
    UI->>API: Send answer request
    API->>API: Validate retrieval configuration
    API->>Store: Look up available article data

    alt Required article data is missing
        API->>Wiki: Retrieve article content
        Wiki-->>API: Article content
        API->>API: Process and chunk content
        API->>Ollama: Generate embeddings
        Ollama-->>API: Embeddings
        API->>Store: Persist documents, chunks, and embeddings
    end

    API->>Store: Retrieve lexical and vector candidates
    Store-->>API: Candidate passages
    API->>API: Rank and prepare evidence
    API->>Ollama: Generate answer using evidence
    Ollama-->>API: Generated answer
    API->>Verify: Validate and format answer
    Verify-->>API: Clean answer and final article link
    API->>Store: Persist conversation turn
    API-->>UI: Return answer or stream response
    UI-->>User: Display answer and final citation
```

The diagrams describe the intended documented flow. Exact internal sequencing can vary with cache hits, streaming behavior, and the implementation of individual services.

---

## 5. How a question is processed

The following is the end-to-end workflow described for Nexia.

1. **Question submission**  
   The user enters a question in the React interface. The frontend sends it to the FastAPI answer endpoint, either as a normal request or through the streaming endpoint.

2. **Request validation and retrieval configuration**  
   The backend validates the request and applies the configured retrieval limits.

3. **Question analysis**  
   The backend analyzes the question to identify its intent, important entities, relationships, or sub-questions. The query-planning layer can decompose complex questions when needed.

4. **Knowledge lookup**  
   The backend checks whether the relevant Wikipedia article content is already available in the local database.

5. **Wikipedia retrieval and ingestion**  
   If required content is unavailable, the backend retrieves the relevant article content, processes it, divides it into passages, and prepares embeddings.

6. **Persistence**  
   Article data, passages, and embeddings are stored in PostgreSQL and pgvector for subsequent retrieval.

7. **Hybrid retrieval**  
   The retrieval layer combines lexical BM25 matching with vector similarity search to identify relevant passages.

8. **Evidence ranking and packing**  
   Candidate passages are ranked, filtered, and packed into a bounded evidence context.

9. **Grounded prompt construction**  
   The backend constructs a prompt containing the question and selected evidence.

10. **Local answer generation**  
    Ollama runs the configured language model to generate an answer from the supplied context.

11. **Validation and response formatting**  
    The backend applies its answer-validation and citation-formatting logic. Internal evidence markers should not appear in the user-facing prose.

12. **Final article citation**  
    When valid source metadata is available, the response includes one relevant Wikipedia article hyperlink at the end of the answer.

13. **Delivery and persistence**  
    The backend returns the completed answer or streams it to the frontend. Conversation data is persisted so that it can be retrieved later.

---

## 6. Repository layout

The following is the repository structure described in the supplied project documentation. Update this tree if the actual repository has changed.

```text
Nexia/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── middleware/
│   │   │   └── routes/
│   │   ├── core/
│   │   ├── db/
│   │   ├── retrieval/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   ├── deploy/
│   ├── tests/
│   │   └── test_nexia.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── config/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── stores/
│   │   ├── test/
│   │   ├── types/
│   │   ├── utils/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── vite.config.ts
├── run_tests.py
├── GITHUB_ISSUES_BACKLOG.md
├── .gitignore
└── README.md
```

### Important directories

| Path | Purpose |
|---|---|
| `backend/app/api/` | HTTP routes and middleware |
| `backend/app/core/` | Application settings and shared infrastructure |
| `backend/app/db/` | Database models, connections, and persistence |
| `backend/app/retrieval/` | Question analysis and retrieval planning |
| `backend/app/schemas/` | Pydantic request and response schemas |
| `backend/app/services/` | Retrieval, inference, orchestration, and citation services |
| `frontend/src/components/` | Chat interface and reusable UI components |
| `frontend/src/config/` | Frontend configuration |
| `frontend/src/hooks/` | Reusable React hooks |
| `frontend/src/services/` | Backend API and streaming clients |
| `frontend/src/stores/` | Client-side state |
| `frontend/src/test/` | Frontend test setup and test files |

---

## 7. Prerequisites

Install the following before setting up Nexia.

| Requirement | Documented version | Purpose |
|---|---:|---|
| Python | 3.12+ | Backend runtime |
| Node.js | 18+ | Frontend tooling |
| npm | Version compatible with the installed Node.js | Frontend dependencies |
| PostgreSQL | 15+ | Persistent storage |
| pgvector | 0.3+ | Vector storage and similarity search |
| Ollama | 0.4+ | Local model runtime |
| Git | Recent version | Source control |

The documented local model setup uses `gemma3:4b` and `nomic-embed-text`. Model memory and inference speed depend on the machine and the Ollama configuration.

---

## 8. Installation and setup

### 8.1 Clone the repository

```bash
git clone https://github.com/Deekshith1983/Nexia.git
cd Nexia
```

### 8.2 Set up PostgreSQL and pgvector

Install PostgreSQL and ensure the pgvector extension is available to your PostgreSQL installation.

Create the database:

```sql
CREATE DATABASE nexia;
```

Connect to the database and enable the vector extension:

```sql
\c nexia
CREATE EXTENSION IF NOT EXISTS vector;
```

Use the database initialization or migration process provided by the backend. Do not assume that creating the database alone creates all application tables.

### 8.3 Install Ollama and download models

Install [Ollama](https://ollama.com/), then download the documented models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

Check that the models are available:

```bash
ollama list
```

The default Ollama service is commonly available at:

```text
http://localhost:11434
```

### 8.4 Set up the backend

Create a virtual environment from the repository root:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

If the configured NLP pipeline requires the spaCy English model, install it:

```bash
python -m spacy download en_core_web_sm
```

Create the local environment file:

**Windows PowerShell**

```powershell
Copy-Item backend/.env.example backend/.env
```

**Linux/macOS**

```bash
cp backend/.env.example backend/.env
```

Review `backend/.env.example` and configure the values required by the current backend settings. The following illustrates the principal connection settings; use the exact variable names expected by the application:

```env
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/nexia
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=gemma3:4b
EMBEDDING_MODEL=nomic-embed-text
```

Replace the example database credentials with your own local values. Never commit a real `.env` file or credentials.

### 8.5 Set up the frontend

From the repository root:

```bash
cd frontend
npm install
cd ..
```

Use the package scripts defined in `frontend/package.json` for development, testing, and production builds.

---

## 9. Run Nexia

Start PostgreSQL and Ollama before starting the application.

### 9.1 Verify Ollama models

```bash
ollama list
```

Confirm that `gemma3:4b` and `nomic-embed-text` are available.

### 9.2 Start the backend

Activate the Python virtual environment, then run the FastAPI application from the repository root.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "backend"
uvicorn app.main:app --app-dir backend --reload --port 8000
```

**Linux/macOS**

```bash
source .venv/bin/activate
PYTHONPATH=backend uvicorn app.main:app --app-dir backend --reload --port 8000
```

Backend URLs documented for the application:

- Health check: `http://localhost:8000/health`
- Interactive API documentation: `http://localhost:8000/docs`



### 9.3 Start the frontend

Open another terminal:

```bash
cd frontend
npm run dev
```

Open the local URL printed by Vite. The documented default is:

```text
http://localhost:5173
```

---

## 10. Testing

The supplied project documentation describes test runner intended to run backend tests, frontend tests, and the frontend production build.


The documented test commands are:

**Backend**

```bash
pytest backend/tests/test_nexia.py -v
```

**Frontend**

```bash
npm --prefix frontend test
```

**Frontend production build**

```bash
npm --prefix frontend run build
```

### Areas to verify

- Backend startup and health endpoint.
- Conversation operations and message persistence.
- Question analysis and retrieval behavior.
- BM25 and vector retrieval integration.
- Evidence selection and passage limits.
- Retrieval configuration enforcement.
- Answer generation and response formatting.
- Suppression of internal citation markers.
- Final article hyperlink formatting.
- Streaming event handling.
- Frontend chat and conversation behavior.
- Frontend TypeScript compilation and production build.

Some integration tests may require PostgreSQL, pgvector, Ollama, downloaded models, or network access to Wikipedia. Configure a dedicated test environment and avoid pointing tests at production data.

> **Test inventory note:** A single command to run the project tests is different from having only one physical test file. The runner may invoke separate backend and frontend test suites because they use different runtimes and test frameworks. Keep the test organization that exists in the repository and use the unified command for full verification.

---

## 11. API overview

The following routes are documented as frontend-facing application endpoints. Check the live FastAPI OpenAPI page at `http://localhost:8000/docs` for the authoritative request and response schemas.

All versioned application routes use the `/api/v1` prefix unless otherwise noted.

### Health and metrics

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check backend availability |
| `GET` | `/api/v1/metrics` | Retrieve application metrics, if enabled |

### Conversations

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/conversations` | List conversations |
| `POST` | `/api/v1/conversations` | Create a conversation |
| `GET` | `/api/v1/conversations/{conversation_id}` | Retrieve a conversation |
| `PATCH` | `/api/v1/conversations/{conversation_id}` | Update or rename a conversation |
| `DELETE` | `/api/v1/conversations/{conversation_id}` | Delete a conversation |

### Answer generation

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/retrieval/answer` | Request a non-streaming answer |
| `POST` | `/api/v1/retrieval/answer/stream` | Request an answer using SSE streaming |
| `POST` | `/api/v1/retrieval/cancel/{request_id}` | Request cancellation of an active generation |

The answer endpoints use the backend's request schema. Consult `/docs` for the exact fields and response structures rather than relying on assumptions in client code.

---


## 12. Engineering decisions

### Retrieval and generation are separate responsibilities

The retrieval layer finds and ranks source material. The language model generates a response using the prepared evidence. This separation makes it possible to inspect and improve retrieval without replacing the answer-generation component.

### Wikipedia is the configured knowledge source

The documented workflow retrieves article content from Wikipedia and stores processed material locally for subsequent use. The scope of answers is therefore bounded by the retrieved content and the coverage of the configured source.

### Local model inference

Ollama provides the model runtime for answer generation and embeddings. This avoids requiring a hosted LLM API for the documented local workflow. Wikipedia retrieval still requires network access when article content is not already available locally.

### One final article link

The user-facing answer is intended to contain clean prose followed by one relevant Wikipedia article hyperlink at the end. The link should come from retrieved article metadata, not from an invented model-generated URL.

### Shared answer preparation

Normal and streaming answer paths should use the same question analysis, retrieval, evidence preparation, and validation logic. Only the delivery mechanism should differ.

---

## 13. Known limitations and future work

The following are areas to assess as the project evolves. They are not claims that these capabilities are already implemented.

- **Knowledge-source coverage:** Wikipedia coverage and article quality determine what evidence can be retrieved.
- **Local inference performance:** Generation speed depends on hardware, model size, context length, and Ollama configuration.
- **Single-instance coordination:** In-memory queues or rate limits, if used, apply only within the process that owns them. Multi-worker deployments need shared coordination.
- **Retrieval quality:** Complex, ambiguous, or multi-hop questions may require further improvements to query planning, ranking, and evidence coverage.
- **Operational monitoring:** Production deployments may benefit from more extensive metrics, tracing, and alerting.
- **Deployment hardening:** Reverse proxy, TLS, process supervision, backups, and resource limits should be configured for any public deployment.

---

## 14. Contributing

Contributions should preserve Nexia's existing architecture and evidence-grounding principles.

### Suggested workflow

1. Fork the repository.
2. Clone your fork.
3. Create a focused branch.
4. Set up the development environment using this README.
5. Make a focused change.
6. Add or update tests.
7. Run the relevant tests and the unified test command.
8. Commit with a clear message.
9. Open a pull request describing the change and verification performed.

Example:

```bash
git checkout -b feature/describe-your-change
git add .
git commit -m "feat(retrieval): improve evidence ranking"
git push origin feature/describe-your-change
```

In a pull request, include:
- The problem being addressed.
- The implementation approach.
- Relevant screenshots or API examples, when useful.
- Tests executed and their results.
- Any configuration or migration requirements.

Do not commit `.env` files, credentials, local databases, generated model files, or unrelated build artifacts.

---

## 15. Development workflow

### Backend development

From the repository root, activate the virtual environment and run:

```bash
uvicorn app.main:app --app-dir backend --reload --port 8000
```

### Frontend development

```bash
npm --prefix frontend run dev
```

### Run backend tests

```bash
pytest backend/tests/test_nexia.py -v
```

### Run frontend tests

```bash
npm --prefix frontend test
```

### Build the frontend

```bash
npm --prefix frontend run build
```

### Run full project verification

```bash
python run_tests.py
```

Use the backend Swagger UI at `http://localhost:8000/docs` to inspect the API contracts while developing frontend integrations.

---

## 16. License

Nexia is licensed under the MIT License.

You are free to use, copy, modify, publish, distribute, sublicense, and sell copies of this software, subject to the conditions of the MIT License

---

## 17. Author and attribution

**Nexia was created and developed from scratch by its original author.**

- **Creator and Lead Developer:** Deekshith
- **Email:** dikideeku@gmail.com`
- **GitHub:** [@Deekshith1983](https://github.com/Deekshith1983)
- **Repository:** [Deekshith1983/Nexia](https://github.com/Deekshith1983/Nexia)

The project’s original architecture, application code, retrieval workflows, evidence preparation, interface, and integrations are attributed to the author, subject to the licenses and attribution requirements of third-party dependencies and tools.

---

*Built with a focus on grounded answers, transparent evidence handling, and local AI inference.*
