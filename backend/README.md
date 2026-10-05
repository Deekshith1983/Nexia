# Nexia Backend

FastAPI backend powering the Nexia Intelligent Knowledge Retrieval system.

## Setup & Running

```bash
# From the project root
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --app-dir backend --reload --port 8000
```

## Running Tests

```bash
pytest backend/tests
```
