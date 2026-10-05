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
├── .gitignore
└── README.md
```

## Quick Start (Stage 0)

1. **Activate Virtual Environment**:
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

2. **Run Server**:
   ```bash
   uvicorn app.main:app --app-dir backend --reload --port 8000
   ```

3. **Verify Health Endpoint**:
   - `GET http://localhost:8000/health` -> `{"status": "ok"}`
   - `GET http://localhost:8000/docs` -> Interactive Swagger UI
