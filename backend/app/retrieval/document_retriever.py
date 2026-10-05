"""Document Retriever Orchestrator for Stage 1 Single Document Retrieval."""

from typing import Optional, List
from app.schemas.retrieval import (
    SearchResponse,
    SearchCandidate,
    DocumentRetrievalResponse,
    Document,
)
from app.retrieval.wikipedia_client import WikipediaClient, WikipediaClientError
from app.retrieval.parser import ArticleParser


class DocumentRetriever:
    """Coordinates Wikipedia search, candidate selection, article acquisition, and document parsing."""

    def __init__(self, wikipedia_client: Optional[WikipediaClient] = None):
        self.client = wikipedia_client or WikipediaClient()
        self.parser = ArticleParser()

    async def search_candidates(self, query: str, limit: int = 10) -> SearchResponse:
        """Search Wikipedia and return structured candidate articles."""
        clean_query = query.strip()
        if not clean_query:
            return SearchResponse(query=query, total_hits=0, candidates=[])

        candidates = await self.client.search(clean_query, limit=limit)
        return SearchResponse(
            query=clean_query,
            total_hits=len(candidates),
            candidates=candidates,
        )

    async def retrieve_document(
        self, query: str, explicit_title: Optional[str] = None
    ) -> DocumentRetrievalResponse:
        """Retrieve and parse a full Wikipedia document based on query or title."""
        target_query = query.strip()
        
        # If explicit title is provided, fetch directly
        if explicit_title and explicit_title.strip():
            title_to_fetch = explicit_title.strip()
            try:
                page_data = await self.client.fetch_page_content(title_to_fetch)
                document = self.parser.parse_raw_page(page_data)
                
                selected_candidate = SearchCandidate(
                    title=document.title,
                    page_id=document.page_id,
                    snippet=document.introduction[:200] if document.introduction else "",
                    url=document.url,
                    word_count=document.metadata.total_words,
                )

                return DocumentRetrievalResponse(
                    query=target_query or title_to_fetch,
                    status="success",
                    selected_candidate=selected_candidate,
                    document=document,
                    other_candidates=[],
                )
            except WikipediaClientError as exc:
                return DocumentRetrievalResponse(
                    query=target_query or title_to_fetch,
                    status="not_found" if "not found" in str(exc).lower() else "error",
                    error_message=str(exc),
                )

        if not target_query:
            return DocumentRetrievalResponse(
                query="",
                status="error",
                error_message="Query string cannot be empty.",
            )

        try:
            # 1. Search candidates
            candidates = await self.client.search(target_query, limit=8)
            if not candidates:
                return DocumentRetrievalResponse(
                    query=target_query,
                    status="not_found",
                    error_message=f"No Wikipedia articles found for query: '{target_query}'",
                )

            # 2. Select best candidate
            selected, others = self._select_candidate(target_query, candidates)

            # 3. Fetch full page content
            page_data = await self.client.fetch_page_content(selected.title)

            # 4. Parse into structured Document
            document = self.parser.parse_raw_page(page_data)

            return DocumentRetrievalResponse(
                query=target_query,
                status="success",
                selected_candidate=selected,
                document=document,
                other_candidates=others,
            )

        except WikipediaClientError as exc:
            return DocumentRetrievalResponse(
                query=target_query,
                status="error",
                error_message=str(exc),
            )
        except Exception as exc:
            return DocumentRetrievalResponse(
                query=target_query,
                status="error",
                error_message=f"Unexpected retrieval failure: {exc}",
            )

    def _select_candidate(
        self, query: str, candidates: List[SearchCandidate]
    ) -> tuple[SearchCandidate, List[SearchCandidate]]:
        """Select the most relevant candidate article while preserving the rest for inspection."""
        normalized_query = query.strip().lower()

        # Check for exact title match
        for idx, cand in enumerate(candidates):
            if cand.title.strip().lower() == normalized_query:
                selected = cand
                others = candidates[:idx] + candidates[idx + 1 :]
                return selected, others

        # Default to top search rank
        selected = candidates[0]
        others = candidates[1:]
        return selected, others
