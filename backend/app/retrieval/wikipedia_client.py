"""Asynchronous Wikipedia API Client."""

import urllib.parse
from typing import List, Dict, Any, Optional
import httpx

from app.schemas.retrieval import SearchCandidate


class WikipediaClientError(Exception):
    """Base exception for Wikipedia client operations."""
    pass


class WikipediaClient:
    """Client for querying and fetching articles from the Wikipedia Action API."""

    BASE_URL = "https://en.wikipedia.org/w/api.php"
    USER_AGENT = "NexiaBot/0.1.0 (https://github.com/nexia-retrieval; contact@nexia.local)"

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json",
        }

    async def search(self, query: str, limit: int = 10) -> List[SearchCandidate]:
        """Search Wikipedia for candidate articles matching the query."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "utf8": 1,
        }

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            try:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                raise WikipediaClientError(f"Wikipedia search request failed: {exc}") from exc
            except Exception as exc:
                raise WikipediaClientError(f"Unexpected error during Wikipedia search: {exc}") from exc

        query_data = data.get("query", {})
        search_results = query_data.get("search", [])

        candidates: List[SearchCandidate] = []
        for item in search_results:
            title = item.get("title", "")
            page_id = item.get("pageid", 0)
            snippet = item.get("snippet", "")
            word_count = item.get("wordcount")

            # Clean HTML tags like <span class="searchmatch"> from snippet
            clean_snippet = snippet.replace('<span class="searchmatch">', "").replace("</span>", "")

            # Build canonical URL
            encoded_title = urllib.parse.quote(title.replace(" ", "_"))
            url = f"https://en.wikipedia.org/wiki/{encoded_title}"

            candidates.append(
                SearchCandidate(
                    title=title,
                    page_id=page_id,
                    snippet=clean_snippet,
                    url=url,
                    word_count=word_count,
                )
            )

        return candidates

    async def fetch_page_content(self, title_or_pageid: str) -> Dict[str, Any]:
        """Fetch full extracted plaintext and metadata for a Wikipedia article.
        
        Uses explaintext=1 and exsectionformat=wiki to get structured wiki headings
        and pure paragraph text without HTML tags.
        """
        params: Dict[str, Any] = {
            "action": "query",
            "prop": "extracts|info|categories",
            "inprop": "url",
            "explaintext": 1,
            "exsectionformat": "wiki",
            "format": "json",
            "utf8": 1,
            "redirects": 1,
            "cllimit": 50,
        }

        # Check if title_or_pageid is numeric pageid or string title
        if title_or_pageid.isdigit():
            params["pageids"] = title_or_pageid
        else:
            params["titles"] = title_or_pageid

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            try:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                raise WikipediaClientError(f"Wikipedia page fetch request failed: {exc}") from exc
            except Exception as exc:
                raise WikipediaClientError(f"Unexpected error during Wikipedia page fetch: {exc}") from exc

        query_data = data.get("query", {})
        pages = query_data.get("pages", {})

        if not pages:
            raise WikipediaClientError(f"No page data returned for: {title_or_pageid}")

        # Wikipedia returns pages dict keyed by page ID
        first_page_id = next(iter(pages))
        page_info = pages[first_page_id]

        if "missing" in page_info or int(first_page_id) < 0:
            raise WikipediaClientError(f"Wikipedia article not found for: {title_or_pageid}")

        return page_info
