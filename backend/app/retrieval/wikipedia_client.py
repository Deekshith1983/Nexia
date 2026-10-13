"""Asynchronous Wikipedia API Client."""

import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
from html.parser import HTMLParser
import httpx

from app.schemas.retrieval import SearchCandidate


class WikipediaClientError(Exception):
    """Base exception for Wikipedia client operations."""
    pass


class WikiTableParser(HTMLParser):
    """Extracts structured rows and internal hyperlinks from Wikipedia HTML wikitables."""

    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row: List[str] = []
        self.current_cell: List[str] = []
        self.tables: List[List[List[str]]] = []
        self.current_table: List[List[str]] = []
        self.links: List[Tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if tag == "table":
            if "wikitable" in attr_dict.get("class", ""):
                self.in_table = True
                self.current_table = []
        elif self.in_table:
            if tag == "tr":
                self.in_row = True
                self.current_row = []
            elif tag in ("td", "th") and self.in_row:
                self.in_cell = True
                self.current_cell = []
            elif tag == "a" and self.in_cell:
                title = attr_dict.get("title", "")
                href = attr_dict.get("href", "")
                if title and not title.startswith("Help:") and not title.startswith("File:") and not title.startswith("Wikipedia:"):
                    self.links.append((title, href))

    def handle_endtag(self, tag):
        if tag == "table" and self.in_table:
            self.in_table = False
            if self.current_table:
                self.tables.append(self.current_table)
        elif self.in_table:
            if tag == "tr" and self.in_row:
                self.in_row = False
                if self.current_row:
                    self.current_table.append(self.current_row)
            elif tag in ("td", "th") and self.in_cell:
                self.in_cell = False
                text = " ".join(self.current_cell).strip()
                if text:
                    self.current_row.append(text)

    def handle_data(self, data):
        if self.in_cell:
            text = data.strip()
            if text:
                self.current_cell.append(text)

    @classmethod
    def format_tables_as_paragraphs(cls, html_content: str) -> Tuple[str, List[str]]:
        """Convert HTML wikitables into clean formatted text paragraphs and unique entity links."""
        parser = cls()
        parser.feed(html_content)
        paragraphs: List[str] = []

        for table in parser.tables:
            if not table or len(table) < 2:
                continue
            headers = table[0]
            for row in table[1:]:
                if len(row) >= 2:
                    row_parts = []
                    for col_idx, cell_val in enumerate(row):
                        if col_idx < len(headers):
                            h = headers[col_idx]
                            if h and not h.isdigit() and len(h) < 30:
                                row_parts.append(f"{h}: {cell_val}")
                            else:
                                row_parts.append(cell_val)
                        else:
                            row_parts.append(cell_val)
                    para_text = " | ".join(row_parts)
                    if len(para_text) > 20:
                        paragraphs.append(para_text)

        unique_links = []
        for title, _ in parser.links:
            if title not in unique_links and len(title) > 2:
                unique_links.append(title)

        return "\n\n".join(paragraphs), unique_links


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

        # Check if table parsing is needed (e.g. list pages, tables stripped by explaintext)
        title = page_info.get("title", "")
        raw_extract = page_info.get("extract", "")
        needs_table = (
            title.lower().startswith("list of")
            or "== list ==" in raw_extract.lower()
            or len(raw_extract) < 1500
        )

        if needs_table:
            try:
                parse_params = {
                    "action": "parse",
                    "page": title,
                    "prop": "text",
                    "format": "json",
                }
                async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as parse_client:
                    p_res = await parse_client.get(self.BASE_URL, params=parse_params)
                    if p_res.status_code == 200:
                        html_text = p_res.json().get("parse", {}).get("text", {}).get("*", "")
                        if html_text:
                            table_text, table_links = WikiTableParser.format_tables_as_paragraphs(html_text)
                            if table_text:
                                page_info["extract"] = raw_extract + "\n\n== Tabular Details ==\n\n" + table_text
                                page_info["extracted_links"] = table_links
            except Exception:
                pass  # Fall back to base extract if HTML parsing encounters error

        return page_info
