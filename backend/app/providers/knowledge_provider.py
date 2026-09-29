"""
Nexia Knowledge Provider Boundary.
Abstracts external knowledge acquisition behind a provider interface.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.core.errors import KnowledgeProviderError
from app.services.wikipedia_retriever import RetrievalResult, RetrievalStatus, wikipedia_retriever
from app.services.wikipedia_navigator import (
    NavigationStep,
    WikipediaArticle,
    WikipediaSection,
    wikipedia_navigator,
)


class KnowledgeProvider(ABC):
    """Abstract interface defining external knowledge retrieval and navigation."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        section_name: Optional[str] = None,
        context_hint: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve factual content for a query or section."""
        pass

    @abstractmethod
    async def inspect_structure(self, page_title: str) -> WikipediaArticle:
        """Inspect article sections, internal links, and redirect/disambiguation status."""
        pass

    @abstractmethod
    async def retrieve_section(self, page_title: str, section_title: str) -> RetrievalResult:
        """Retrieve content from a specific section of an article."""
        pass

    @abstractmethod
    async def health(self) -> bool:
        """Verify knowledge provider reachability."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider identifier."""
        pass


class WikipediaKnowledgeProvider(KnowledgeProvider):
    """Production Wikipedia knowledge provider implementation."""

    def __init__(self):
        self._retriever = wikipedia_retriever
        self._navigator = wikipedia_navigator

    async def retrieve(
        self,
        query: str,
        section_name: Optional[str] = None,
        context_hint: Optional[str] = None,
    ) -> RetrievalResult:
        try:
            return await self._retriever.retrieve(
                query=query,
                section_name=section_name,
                context_hint=context_hint,
            )
        except Exception as e:
            raise KnowledgeProviderError(f"Wikipedia retrieval failed: {e}") from e

    async def inspect_structure(self, page_title: str) -> WikipediaArticle:
        try:
            return await self._navigator.retrieve_article_structure(page_title)
        except Exception as e:
            raise KnowledgeProviderError(f"Wikipedia structure inspection failed: {e}") from e

    async def retrieve_section(self, page_title: str, section_title: str) -> RetrievalResult:
        try:
            return await self._navigator.retrieve_section(page_title, section_title)
        except Exception as e:
            raise KnowledgeProviderError(f"Wikipedia section retrieval failed: {e}") from e

    async def health(self) -> bool:
        try:
            res = await self._retriever.retrieve(query="Test")
            return res.status in (RetrievalStatus.SUCCESS, RetrievalStatus.NOT_FOUND, RetrievalStatus.DISAMBIGUATION)
        except Exception:
            return False

    def get_provider_name(self) -> str:
        return "wikipedia"


class MockKnowledgeProvider(KnowledgeProvider):
    """Deterministic mock knowledge provider for hermetic testing."""

    def __init__(self):
        self._store: Dict[str, RetrievalResult] = {}
        self._healthy = True

    def set_content(self, query: str, content: str, title: Optional[str] = None) -> None:
        title = title or query
        self._store[query.lower()] = RetrievalResult(
            query=query,
            status=RetrievalStatus.SUCCESS,
            page_title=title,
            page_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            content=content,
            source="wikipedia",
        )

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    async def retrieve(
        self,
        query: str,
        section_name: Optional[str] = None,
        context_hint: Optional[str] = None,
    ) -> RetrievalResult:
        if not self._healthy:
            raise KnowledgeProviderError("Mock knowledge provider configured as unhealthy.")
        key = query.lower()
        if key in self._store:
            return self._store[key]
        return RetrievalResult(
            query=query,
            status=RetrievalStatus.NOT_FOUND,
            error_message="Not found in mock store",
            source="wikipedia",
        )

    async def inspect_structure(self, page_title: str) -> WikipediaArticle:
        if not self._healthy:
            raise KnowledgeProviderError("Mock knowledge provider configured as unhealthy.")
        ret = await self.retrieve(page_title)
        return WikipediaArticle(
            page_title=ret.page_title or page_title,
            canonical_url=ret.page_url or f"https://en.wikipedia.org/wiki/{page_title}",
            summary=ret.content[:200],
            sections=[WikipediaSection(index=1, title="Overview", content=ret.content)],
            links=[],
        )

    async def retrieve_section(self, page_title: str, section_title: str) -> RetrievalResult:
        return await self.retrieve(page_title)

    async def health(self) -> bool:
        return self._healthy

    def get_provider_name(self) -> str:
        return "mock-wikipedia"


# Singleton provider registry
_active_knowledge_provider: Optional[KnowledgeProvider] = None


def get_knowledge_provider() -> KnowledgeProvider:
    """Return the active singleton knowledge provider."""
    global _active_knowledge_provider
    if _active_knowledge_provider is None:
        _active_knowledge_provider = WikipediaKnowledgeProvider()
    return _active_knowledge_provider


def set_knowledge_provider(provider: KnowledgeProvider) -> None:
    """Override active knowledge provider."""
    global _active_knowledge_provider
    _active_knowledge_provider = provider
