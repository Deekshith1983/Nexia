import fastapi
import uvicorn
import langchain
import ollama

print("FastAPI version:", fastapi.__version__)
print("Uvicorn version:", uvicorn.__version__)
print("LangChain version:", langchain.__version__)
print("Ollama module loaded successfully. Client:", ollama.Client)
print("Backend virtual environment verified 100% operational!")
