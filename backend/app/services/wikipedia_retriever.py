import logging
import re
import unicodedata
import urllib.parse
from typing import Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("nexia.wiki")


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


# ── Normalization & scoring ───────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


def _clean_extract(text: str) -> str:
    """
    Remove Wikipedia markup artifacts from a REST API extract before
    passing the content to the LLM.

    Specifically handles:
    - [O.S. 15 December 1642] — Old Style (Julian) calendar annotations
    - [N.S. 4 January 1643]   — New Style (Gregorian) calendar annotations
    - [citation needed], [nb 1], [note ...] — editorial annotations
    - [1], [2], [12] — inline citation superscripts
    - Duplicate whitespace introduced by removals

    Example:
        "Sir Isaac Newton (25 December 1642 [O.S. 15 December 1642] – 20 March 1727)"
        → "Sir Isaac Newton (25 December 1642 – 20 March 1727)"
    """
    # Remove [O.S. ...] and [N.S. ...] calendar style annotations
    text = re.sub(r"\[(?:O\.S\.|N\.S\.|Old\s+Style|New\s+Style)[^\]]*\]", "", text)
    # Remove other common editorial bracketed annotations
    # (citation needed, nb N, note N, clarification needed, etc.)
    text = re.sub(
        r"\[(?:citation\s+needed|clarification\s+needed|nb\s+\d+|note\s+\d+)[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove pure numeric citation superscripts like [1], [12], [123]
    text = re.sub(r"\[\d{1,3}\]", "", text)
    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _title_case(text: str) -> str:
    """Convert 'narendra damodardas modi' -> 'Narendra Damodardas Modi'."""
    return " ".join(w.capitalize() for w in text.split())


def _word_set(text: str) -> set:
    """Return set of meaningful words (3+ chars) from a normalized string."""
    stop = {"the", "a", "an", "of", "in", "on", "at", "to", "for",
            "is", "was", "are", "were", "who", "what", "when", "where", "how"}
    return {w for w in _normalize(text).split() if len(w) >= 3 and w not in stop}


def _score_candidate(candidate_title: str, entity: Optional[str], query: str) -> int:
    """
    Score a Wikipedia search candidate against the target entity / query.

    Scoring tiers (highest wins):
      200 — exact normalized title match with entity
      180 — title with trailing parenthetical stripped == entity
      160 — bidirectional substring (essentially same string)
      140 — title starts with entity
      120 — entity starts with title
       80 — all words of title appear in entity words (e.g. "Narendra Modi" ⊆ words of "narendra damodardas modi")
       40 — majority of title words appear in entity words (> 50%)
       10 — entity appears anywhere in title as whole word
        0 — no structural match
    """
    norm_title = _normalize(candidate_title)

    if entity:
        norm_entity = _normalize(entity)

        # Tier 200 — exact match
        if norm_title == norm_entity:
            return 200

        # Tier 180 — title with trailing disambiguation stripped
        title_stripped = re.sub(r"\s*\(.*?\)\s*$", "", norm_title).strip()
        if title_stripped == norm_entity:
            return 180

        # Tier 160 — bidirectional substring
        if norm_entity in norm_title and norm_title in norm_entity:
            return 160

        # Tier 140 — title starts with entity
        if norm_title.startswith(norm_entity):
            return 140

        # Tier 120 — entity starts with title (title is a prefix of entity)
        if norm_entity.startswith(norm_title):
            return 120

        # Tier 80/40 — word-set overlap
        # "Narendra Modi" words = {"narendra", "modi"}
        # entity "narendra damodardas modi" words = {"narendra", "damodardas", "modi"}
        # → all title words appear in entity words → score 80
        title_words = _word_set(candidate_title)
        entity_words = _word_set(entity)
        if title_words and entity_words:
            overlap = title_words & entity_words
            ratio = len(overlap) / len(title_words)
            if ratio == 1.0:
                return 80
            elif ratio > 0.5:
                return 40

        # Tier 10 — loose whole-word containment
        try:
            if re.search(r"\b" + re.escape(norm_entity) + r"\b", norm_title):
                return 10
        except re.error:
            if norm_entity in norm_title:
                return 10

    return 0


def _select_best_candidate(
    search_results: List[Dict],
    entity: Optional[str],
    query: str,
) -> str:
    """Pick the best Wikipedia candidate using entity-aware ranking."""
    if not search_results:
        raise ValueError("Empty search results")

    scored: List[Tuple[int, int, str]] = []  # (score, index, title)
    for idx, result in enumerate(search_results):
        title = result.get("title", "")
        if not title:
            continue
        score = _score_candidate(title, entity, query)
        rank_bonus = max(0, 5 - idx)
        # Disambiguation penalty
        if re.search(r"\(disambiguation\)", title, re.IGNORECASE):
            score = max(0, score - 100)
        scored.append((score + rank_bonus, idx, title))

    if not scored:
        return search_results[0].get("title", "")

    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, best_idx, best_title = scored[0]

    logger.debug(
        "[WIKI] Candidate ranking (entity=%r):\n%s\n  -> selected: %r (score=%d)",
        entity,
        "\n".join(f"    [{i}] score={s}  title={t!r}" for s, i, t in scored),
        best_title,
        best_score,
    )
    return best_title


# ── Retriever ─────────────────────────────────────────────────────────────────

class WikipediaRetriever:
    """Service responsible for querying Wikipedia APIs and retrieving article content."""

    SEARCH_API_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    USER_AGENT = "NexiaAgent/1.0 (https://github.com/Deekshith1983/Nexia)"

    async def _fetch_summary(
        self, client: httpx.AsyncClient, title: str
    ) -> Optional[Dict]:
        """
        Fetch the Wikipedia REST summary for a given page title.
        Returns the parsed JSON dict on success, or None if not found / disambiguation.
        """
        encoded = urllib.parse.quote(title.strip(), safe="")
        try:
            resp = await client.get(f"{self.SUMMARY_API_URL}{encoded}")
        except Exception as e:
            logger.debug("[WIKI] summary fetch error for %r: %s", title, e)
            return None

        if resp.status_code != 200:
            return None

        data = resp.json()
        # Reject disambiguation pages
        if data.get("type") in ("disambiguation",):
            logger.debug("[WIKI] summary for %r is a disambiguation page — skipping", title)
            return None
        extract = data.get("extract", "").strip()
        if not extract:
            return None
        return data

    def _entity_title_variants(self, entity: str) -> List[str]:
        """
        Generate title variants to try for direct Wikipedia lookup.

        Priority order:
          1. Entity as-is (Wikipedia normalises first char to uppercase)
          2. Title-cased version ("narendra damodardas modi" → "Narendra Damodardas Modi")
          3. First + last word only ("Narendra Modi" from "narendra damodardas modi")
             — useful for "full legal name" entities
        """
        raw = entity.strip()
        tc = _title_case(raw)
        variants: List[str] = []

        # Variant 1: as-is
        variants.append(raw)

        # Variant 2: title-cased (skip if same as raw)
        if tc != raw:
            variants.append(tc)

        # Variant 3: first + last word (only if entity has 3+ words)
        words = raw.split()
        if len(words) >= 3:
            short = f"{words[0].capitalize()} {words[-1].capitalize()}"
            if short not in variants:
                variants.append(short)

        return variants

    async def retrieve(
        self,
        query: str,
        entity: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve Wikipedia evidence for a query.

        Strategy (in priority order):
          1. Direct title lookup — try entity name variants. Use the first
             variant that resolves to a real non-disambiguation Wikipedia page.
          2. Full-text search + entity-aware candidate ranking (fallback).
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RetrievalResult(status=RetrievalStatus.NO_RESULTS, query=cleaned_query)

        headers = {"User-Agent": self.USER_AGENT}
        logger.debug("[WIKI] retrieve  query=%r  entity=%r", cleaned_query, entity)

        try:
            async with httpx.AsyncClient(timeout=6.0, headers=headers) as client:

                # ── Strategy 1: Direct entity title lookup ────────────────────
                if entity and entity.strip():
                    for variant in self._entity_title_variants(entity):
                        direct_data = await self._fetch_summary(client, variant)
                        if direct_data:
                            actual_title = direct_data.get("title", variant)
                            extract = _clean_extract(direct_data.get("extract", ""))
                            page_url = (
                                direct_data.get("content_urls", {})
                                .get("desktop", {})
                                .get("page", "")
                            )
                            logger.debug(
                                "[WIKI] direct lookup success  variant=%r -> title=%r",
                                variant, actual_title,
                            )
                            return RetrievalResult(
                                status=RetrievalStatus.SUCCESS,
                                query=cleaned_query,
                                page_title=actual_title,
                                page_url=page_url,
                                content=extract,
                                metadata={
                                    "strategy": "direct_entity_lookup",
                                    "lookup_variant": variant,
                                    "page_id": direct_data.get("pageid"),
                                },
                            )
                    logger.debug(
                        "[WIKI] all direct lookup variants failed for entity=%r; falling back to search",
                        entity,
                    )

                # ── Strategy 2: Full-text search + entity-aware ranking ────────
                search_params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": cleaned_query,
                    "srlimit": 5,
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
                    return RetrievalResult(status=RetrievalStatus.NO_RESULTS, query=cleaned_query)

                logger.debug(
                    "[WIKI] search returned %d candidates: %s",
                    len(search_results),
                    [r.get("title") for r in search_results],
                )

                selected_title = _select_best_candidate(
                    search_results, entity=entity, query=cleaned_query
                )
                if not selected_title:
                    return RetrievalResult(status=RetrievalStatus.NO_RESULTS, query=cleaned_query)

                logger.debug("[WIKI] search selected title=%r", selected_title)

                summary_data = await self._fetch_summary(client, selected_title)
                if not summary_data:
                    return RetrievalResult(
                        status=RetrievalStatus.NO_RESULTS,
                        query=cleaned_query,
                        page_title=selected_title,
                    )

                actual_title = summary_data.get("title", selected_title)
                extract = _clean_extract(summary_data.get("extract", ""))
                page_url = (
                    summary_data.get("content_urls", {})
                    .get("desktop", {})
                    .get("page", "")
                )

                logger.debug(
                    "[WIKI] retrieved  title=%r  url=%s  content_len=%d",
                    actual_title, page_url, len(extract),
                )

                return RetrievalResult(
                    status=RetrievalStatus.SUCCESS,
                    query=cleaned_query,
                    page_title=actual_title,
                    page_url=page_url,
                    content=extract,
                    metadata={
                        "strategy": "search_ranked",
                        "snippet": search_results[0].get("snippet", ""),
                        "page_id": summary_data.get("pageid"),
                        "candidates": [r.get("title") for r in search_results],
                    },
                )

        except (httpx.HTTPError, httpx.TimeoutException, Exception) as e:
            logger.warning("[WIKI] retrieval failed: %s", e)
            return RetrievalResult(
                status=RetrievalStatus.FAILED,
                query=cleaned_query,
                metadata={"error": str(e)},
            )


wikipedia_retriever = WikipediaRetriever()
