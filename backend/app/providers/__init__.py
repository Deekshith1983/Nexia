"""
Nexia Infrastructure Provider Boundaries.
"""
from app.providers.model_provider import (
    LanguageModelProvider,
    OllamaModelProvider,
    MockModelProvider,
    get_model_provider,
    set_model_provider,
)
from app.providers.knowledge_provider import (
    KnowledgeProvider,
    WikipediaKnowledgeProvider,
    MockKnowledgeProvider,
    get_knowledge_provider,
    set_knowledge_provider,
)

__all__ = [
    "LanguageModelProvider",
    "OllamaModelProvider",
    "MockModelProvider",
    "get_model_provider",
    "set_model_provider",
    "KnowledgeProvider",
    "WikipediaKnowledgeProvider",
    "MockKnowledgeProvider",
    "get_knowledge_provider",
    "set_knowledge_provider",
]
