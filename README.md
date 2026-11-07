# Nexia — Intelligent Knowledge Retrieval Backend

Nexia is an intelligent, structured knowledge retrieval and synthesis agent engineered from first principles.

## Development Philosophy
- **Backend-First**: Solidify data acquisition, parsing, structured passage retrieval, and evidence extraction before introducing generative models or frontend interfaces.
- **Traceable Evidence**: Information must be verified and linked from Document → Section → Passage.
- **Strict Staged Roadmap**: Step-by-step verification from Stage 0 through Stage 11.

## Project Structure
```text
Nexia/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/          # API route definitions (e.g., health, retrieve)
│   │   ├── core/                # Global configuration and core settings
│   │   ├── schemas/             # Pydantic data validation and contract models
│   │   ├── services/            # Business logic and external service integrations
│   │   ├── retrieval/           # Knowledge parsing, ranking, and retrieval engine
│   │   └── main.py              # FastAPI application entrypoint
│   ├── tests/                   # Unit and integration test suites
│   ├── requirements.txt         # Python backend dependencies
│   └── README.md                # Backend specific documentation
├── frontend/                    # Responsive React + TypeScript Vite frontend
│   ├── src/                     # React components, hooks, stores, and services
│   ├── package.json             # Frontend dependencies & scripts
│   └── vite.config.ts           # Vite bundler & reverse proxy configuration
├── .gitignore
└── README.md
```

## Quick Start

### 1. Start Backend Server
```bash
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --app-dir backend --reload --port 8000
```
- Health Check: `GET http://localhost:8000/health` -> `{"status": "ok"}`
- Interactive Docs: `GET http://localhost:8000/docs`

### 2. Start Frontend Application
```bash
cd frontend
npm install
npm run dev
```
- Open `http://localhost:5173` in your browser.
