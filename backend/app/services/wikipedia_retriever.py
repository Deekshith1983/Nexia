import urllib.parse
from typing import Dict, Optional
import httpx
from pydantic import BaseModel, Field

class RetrievalStatus:
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    FAILED = "failed"

class RetrievalResult(BaseModel):
    """Structured representation of a Wikipedia retrieval result."""
    status: str  # "success" | "no_results" | "failed"
    query: str
    page_title: Optional[str] = None
    page_url: Optional[str] = None
    content: Optional[str] = None
    source: str = "Wikipedia"
    metadata: Dict = Field(default_factory=dict)

class WikipediaRetriever:
    """Service responsible for querying Wikipedia APIs and retrieving article content."""

    SEARCH_API_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    USER_AGENT = "NexiaAgent/1.0 (https://github.com/Deekshith1983/Nexia)"

    async def retrieve(self, query: str) -> RetrievalResult:
        cleaned_query = query.strip()
        if not cleaned_query:
            return RetrievalResult(
                status=RetrievalStatus.NO_RESULTS,
                query=cleaned_query,
            )

        headers = {"User-Agent": self.USER_AGENT}

        try:
            async with httpx.AsyncClient(timeout=6.0, headers=headers) as client:
                # 1. Search Wikipedia for top matching article title
                search_params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": cleaned_query,
                    "utf8": 1,
                    "format": "json",
                }
                search_res = await client.get(self.SEARCH_API_URL, params=search_params)
                
                if search_res.status_code != 200:
                    return RetrievalResult(
                        status=RetrievalStatus.FAILED,
                        query=cleaned_query,
                        metadata={"http_status": search_res.status_code},
                    )

                search_data = search_res.json()
                search_results = search_data.get("query", {}).get("search", [])

                if not search_results:
                    return RetrievalResult(
                        status=RetrievalStatus.NO_RESULTS,
                        query=cleaned_query,
                    )

                # Select top matching title
                top_title = search_results[0].get("title")
                if not top_title:
                    return RetrievalResult(
                        status=RetrievalStatus.NO_RESULTS,
                        query=cleaned_query,
                    )

                # 2. Fetch summary extract for top title
                encoded_title = urllib.parse.quote(top_title, safe="")
                summary_res = await client.get(f"{self.SUMMARY_API_URL}{encoded_title}")

                if summary_res.status_code == 404:
                    return RetrievalResult(
                        status=RetrievalStatus.NO_RESULTS,
                        query=cleaned_query,
                        page_title=top_title,
                    )
                elif summary_res.status_code != 200:
                    return RetrievalResult(
                        status=RetrievalStatus.FAILED,
                        query=cleaned_query,
                        page_title=top_title,
                        metadata={"http_status": summary_res.status_code},
                    )

                summary_data = summary_res.json()
                extract = summary_data.get("extract", "")
                page_url = summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")

                if not extract.strip():
                    return RetrievalResult(
                        status=RetrievalStatus.NO_RESULTS,
                        query=cleaned_query,
                        page_title=top_title,
                        page_url=page_url,
                    )

                return RetrievalResult(
                    status=RetrievalStatus.SUCCESS,
                    query=cleaned_query,
                    page_title=top_title,
                    page_url=page_url,
                    content=extract.strip(),
                    metadata={
                        "snippet": search_results[0].get("snippet", ""),
                        "page_id": summary_data.get("pageid"),
                    },
                )

        except (httpx.HTTPError, httpx.TimeoutException, Exception) as e:
            print(f"[INFO] Wikipedia retrieval failed ({e}). Returning safe failure result.")
            return RetrievalResult(
                status=RetrievalStatus.FAILED,
                query=cleaned_query,
                metadata={"error": str(e)},
            )

wikipedia_retriever = WikipediaRetriever()
