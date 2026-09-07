# Nexia — Grounded Wikipedia AI Agent

Nexia is a conversational AI agent that provides a ChatGPT/Gemini-like interface while grounding factual claims strictly in Wikipedia evidence.

---

## 🚀 Quick Start Guide

### 1. Backend Setup (FastAPI & LangChain)

```bash
cd backend
# Virtual environment has been created in backend/.venv
.\.venv\Scripts\activate

# Run FastAPI backend server
.\.venv\Scripts\python main.py
```
The backend API will run on `http://127.0.0.1:8000`.

### 2. Frontend Setup (React + Vite)

```bash
cd frontend
# Dependencies installed via npm / pnpm
npm run dev
```
The React chat UI will run on `http://localhost:5173`.

### 3. Local LLM Runtime (Ollama + Qwen3 8B)

1. Download and install [Ollama](https://ollama.com).
2. Pull the Qwen3 8B model:
   ```bash
   ollama run qwen3:8b
   ```
3. Nexia backend will connect to Ollama at `http://localhost:11434`.

---

## 📁 Project Structure

```text
Nexia/
├── backend/
│   ├── .venv/            # Python 3.12 Virtual Environment
│   ├── main.py           # FastAPI Core Application
│   └── requirements.txt  # FastAPI, LangChain, Ollama dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # ChatGPT/Gemini Dark Theme UI
│   │   ├── App.css       # Responsive Glassmorphism Styling
│   │   └── index.css     # CSS Design Tokens & Fonts
│   └── package.json      # React, Vite, Lucide Icons
├── .gitignore
└── README.md
```
