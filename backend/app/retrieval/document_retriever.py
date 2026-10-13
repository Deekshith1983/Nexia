"""Document Retriever Orchestrator with Candidate Scoring & Diagnostic Transparency."""

import re
from typing import Optional, List, Tuple
from app.schemas.retrieval import (
    SearchResponse,
    SearchCandidate,
    DocumentRetrievalResponse,
    Document,
)
from app.retrieval.wikipedia_client import WikipediaClient, WikipediaClientError
from app.retrieval.parser import ArticleParser
from app.retrieval.query_processor import QueryProcessor


class DocumentRetriever:
    """Coordinates Wikipedia search, candidate scoring/disambiguation, article acquisition, and document parsing."""

    def __init__(self, wikipedia_client: Optional[WikipediaClient] = None):
        self.client = wikipedia_client or WikipediaClient()
        self.parser = ArticleParser()
        self.processor = QueryProcessor()

    async def search_candidates(self, query: str, limit: int = 10) -> SearchResponse:
        """Search Wikipedia and return structured candidate articles with selection scores."""
        clean_query = query.strip()
        if not clean_query:
            return SearchResponse(query=query, total_hits=0, candidates=[])

        raw_candidates = await self.client.search(clean_query, limit=limit)
        scored_candidates = self._score_candidates(clean_query, raw_candidates)
        return SearchResponse(
            query=clean_query,
            total_hits=len(scored_candidates),
            candidates=scored_candidates,
        )

    async def retrieve_document(
        self, query: str, explicit_title: Optional[str] = None
    ) -> DocumentRetrievalResponse:
        """Retrieve and parse a full Wikipedia document based on query or explicit title."""
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
                    selection_score=1.0,
                )

                return DocumentRetrievalResponse(
                    query=target_query or title_to_fetch,
                    status="success",
                    selected_candidate=selected_candidate,
                    document=document,
                    other_candidates=[],
                )
            except WikipediaClientError:
                # If direct fetch fails (e.g. title variation), fall back to search below
                if not target_query:
                    return DocumentRetrievalResponse(
                        query=title_to_fetch,
                        status="not_found",
                        error_message=f"Page '{title_to_fetch}' not found on Wikipedia.",
                    )

        if not target_query:
            return DocumentRetrievalResponse(
                query="",
                status="error",
                error_message="Query string cannot be empty.",
            )

        try:
            # 1. Search candidates
            raw_candidates = await self.client.search(target_query, limit=8)
            if not raw_candidates:
                return DocumentRetrievalResponse(
                    query=target_query,
                    status="not_found",
                    error_message=f"No Wikipedia articles found for query: '{target_query}'",
                )

            # 2. Score and select best candidate
            scored_candidates = self._score_candidates(target_query, raw_candidates)
            selected = scored_candidates[0]
            others = scored_candidates[1:]

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

    def _score_candidates(
        self, query: str, candidates: List[SearchCandidate]
    ) -> List[SearchCandidate]:
        """Score candidate titles against query tokens to prefer direct entity matches over administrative sub-offices."""
        query_tokens = set(self.processor.tokenize(query, remove_stopwords=True))
        if not query_tokens:
            query_tokens = set(self.processor.tokenize(query, remove_stopwords=False))

        scored: List[Tuple[SearchCandidate, float]] = []

        for idx, cand in enumerate(candidates):
            title_clean = cand.title.strip().lower()
            title_tokens = set(self.processor.tokenize(cand.title, remove_stopwords=True))
            
            # Base rank penalty (preserves Wikipedia search rank signal as base)
            rank_score = 1.0 / (1.0 + 0.1 * idx)

            # Check exact match
            normalized_query = query.strip().lower()
            if title_clean == normalized_query:
                overlap_score = 2.0
            else:
                # Token Jaccard overlap between query content tokens and title tokens
                intersection = query_tokens.intersection(title_tokens)
                union = query_tokens.union(title_tokens)
                jaccard = len(intersection) / len(union) if union else 0.0

                # Keyword adjustment for secondary administrative offshoots vs list intent
                penalty = 0.0
                is_list_query = any(w in normalized_query for w in ["list", "names", "who are", "who were", "prime ministers", "presidents", "all "])
                offshoot_keywords = {"office", "secretariat", "ministry", "department", "timeline of"}
                for kw in offshoot_keywords:
                    if kw in title_clean and kw not in normalized_query:
                        penalty += 0.25

                if "list of" in title_clean:
                    if is_list_query:
                        penalty -= 0.50  # Boost list articles when query requests lists
                    elif "list of" not in normalized_query:
                        penalty += 0.25

                overlap_score = jaccard - penalty

            total_score = round(rank_score + overlap_score, 4)
            cand.selection_score = total_score
            scored.append((cand, total_score))

        # Sort candidates descending by total selection score
        scored.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in scored]
