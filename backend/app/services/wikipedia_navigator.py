"""
Wikipedia Navigation & Evidence Discovery Service for Nexia Stage 9.

Treats Wikipedia as a structured, navigable knowledge environment:
- Search & Page Selection
- Article Structure & Section Inspection
- Targeted Section Retrieval
- Discovered Link Filtering & Traversal
- Redirect Resolution & Disambiguation Handling
- Bounded Navigation Depth & Cycle Prevention
"""
import logging
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from pydantic import BaseModel, Field

from app.services.wikipedia_retriever import (
    RetrievalResult, RetrievalStatus, _clean_extract, _normalize, _score_candidate,
)

logger = logging.getLogger("nexia.wiki.navigator")


class WikipediaSection(BaseModel):
    """Structured representation of a section within a Wikipedia article."""
    index: int
    title: str
    level: int = 2
    content: str = ""


class WikipediaArticle(BaseModel):
    """Structured article representation containing sections, links, and redirect/disambiguation status."""
    page_title: str
    canonical_url: str
    page_id: Optional[int] = None
    summary: str = ""
    sections: List[WikipediaSection] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    is_disambiguation: bool = False
    disambiguation_options: List[str] = Field(default_factory=list)
    is_redirect: bool = False
    redirect_target: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NavigationStep(BaseModel):
    """A recorded navigation movement within the current research run."""
    action_type: str
    source_page: Optional[str] = None
    target_page: Optional[str] = None
    section: Optional[str] = None
    status: str = "success"
    depth: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NavigationHistory:
    """
    Per-request navigation state tracking to enforce navigation bounds and prevent cycles.
    """

    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        self.steps: List[NavigationStep] = []
        self.visited_pages: Set[str] = set()

    def is_cycle(self, target_page: str) -> bool:
        """Check if navigating to target_page would form a loop."""
        norm_target = _normalize(target_page)
        return norm_target in self.visited_pages

    def can_navigate(self, current_depth: int) -> bool:
        """Check if navigation depth budget allows further traversal."""
        return current_depth < self.max_depth

    def record_step(self, step: NavigationStep) -> None:
        """Log a navigation action and register target page as visited."""
        self.steps.append(step)
        if step.target_page:
            self.visited_pages.add(_normalize(step.target_page))
        if step.source_page:
            self.visited_pages.add(_normalize(step.source_page))

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "step_count": len(self.steps),
            "visited_pages": list(self.visited_pages),
            "max_depth": self.max_depth,
            "steps": [s.model_dump() for s in self.steps],
        }


class WikipediaNavigator:
    """
    Service responsible for fine-grained, structural Wikipedia navigation.
    """

    API_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    USER_AGENT = "NexiaAgent/1.0 (https://github.com/Deekshith1983/Nexia)"

    # Ignored namespaces and meta pages for link discovery
    _META_LINK_PREFIXES = (
        "category:", "file:", "help:", "wikipedia:", "template:", "talk:",
        "portal:", "special:", "user:", "draft:", "timedtext:", "module:",
    )

    async def retrieve_article_structure(
        self,
        title: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> WikipediaArticle:
        """
        Fetch article summary, sections structure, and internal links.
        Resolves redirects and identifies disambiguation pages.
        """
        headers = {"User-Agent": self.USER_AGENT}
        own_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=8.0, headers=headers)
            own_client = True

        try:
            # 1. Fetch summary first for fast summary text & redirect/disambig detection
            summary_data: Optional[Dict] = None
            encoded = urllib.parse.quote(title.strip(), safe="")
            try:
                resp = await client.get(f"{self.SUMMARY_URL}{encoded}")
                if resp.status_code == 200:
                    summary_data = resp.json()
            except Exception as e:
                logger.debug("[NAVIGATOR] summary fetch error: %s", e)

            canonical_title = summary_data.get("title", title) if summary_data else title
            canonical_url = (
                summary_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(canonical_title)}")
                if summary_data else f"https://en.wikipedia.org/wiki/{urllib.parse.quote(canonical_title)}"
            )
            is_disambig = (summary_data.get("type") == "disambiguation") if summary_data else False
            summary_text = _clean_extract(summary_data.get("extract", "")) if summary_data else ""
            page_id = summary_data.get("pageid") if summary_data else None

            # 2. Fetch sections and links via MediaWiki Action API
            parse_params = {
                "action": "parse",
                "page": canonical_title,
                "prop": "sections|links",
                "redirects": 1,
                "format": "json",
            }
            parse_resp = await client.get(self.API_URL, params=parse_params)
            sections: List[WikipediaSection] = []
            links: List[str] = []
            disambig_options: List[str] = []

            if parse_resp.status_code == 200:
                parse_data = parse_resp.json().get("parse", {})
                if not canonical_title or canonical_title == title:
                    canonical_title = parse_data.get("title", canonical_title)
                page_id = page_id or parse_data.get("pageid")

                # Parse sections
                raw_sections = parse_data.get("sections", [])
                for s in raw_sections:
                    sec_title = s.get("line", "").strip()
                    # Strip any html tags in section header
                    sec_title = re.sub(r"<[^>]+>", "", sec_title).strip()
                    if sec_title and sec_title.lower() not in ("see also", "references", "external links", "further reading", "notes"):
                        try:
                            idx = int(s.get("index", 0))
                            lvl = int(s.get("level", 2))
                        except (ValueError, TypeError):
                            idx = len(sections) + 1
                            lvl = 2
                        sections.append(WikipediaSection(index=idx, title=sec_title, level=lvl))

                # Parse links
                raw_links = parse_data.get("links", [])
                for l in raw_links:
                    link_title = l.get("*", "").strip()
                    if link_title and not any(link_title.lower().startswith(p) for p in self._META_LINK_PREFIXES):
                        links.append(link_title)

            # If disambiguation, links are potential candidates
            if is_disambig or re.search(r"\(disambiguation\)", canonical_title, re.IGNORECASE):
                is_disambig = True
                disambig_options = [l for l in links if not re.search(r"\(disambiguation\)", l, re.IGNORECASE)][:10]

            return WikipediaArticle(
                page_title=canonical_title,
                canonical_url=canonical_url,
                page_id=page_id,
                summary=summary_text,
                sections=sections,
                links=links[:50],  # bounded links
                is_disambiguation=is_disambig,
                disambiguation_options=disambig_options,
                is_redirect=(summary_data.get("type") == "redirect") if summary_data else False,
                redirect_target=canonical_title if canonical_title != title else None,
            )

        finally:
            if own_client:
                await client.aclose()

    async def retrieve_section(
        self,
        page_title: str,
        section_title: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> RetrievalResult:
        """
        Targeted retrieval of a specific section within a Wikipedia page.
        """
        headers = {"User-Agent": self.USER_AGENT}
        own_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=8.0, headers=headers)
            own_client = True

        try:
            # 1. Fetch section index
            article = await self.retrieve_article_structure(page_title, client=client)
            norm_sec = _normalize(section_title)
            
            matched_section: Optional[WikipediaSection] = None
            for s in article.sections:
                if norm_sec == _normalize(s.title) or norm_sec in _normalize(s.title) or _normalize(s.title) in norm_sec:
                    matched_section = s
                    break

            if not matched_section:
                logger.debug("[NAVIGATOR] Section %r not found on page %r, returning summary", section_title, page_title)
                return RetrievalResult(
                    status=RetrievalStatus.SUCCESS if article.summary else RetrievalStatus.NO_RESULTS,
                    query=f"{page_title} - {section_title}",
                    page_title=article.page_title,
                    page_url=article.canonical_url,
                    content=article.summary,
                    metadata={"strategy": "section_fallback_summary", "requested_section": section_title},
                )

            # 2. Fetch specific section text via Action API
            params = {
                "action": "parse",
                "page": article.page_title,
                "section": matched_section.index,
                "prop": "text",
                "format": "json",
            }
            resp = await client.get(self.API_URL, params=params)
            if resp.status_code == 200:
                raw_html = resp.json().get("parse", {}).get("text", {}).get("*", "")
                # Strip html tags for plain text extract
                text_clean = re.sub(r"<[^>]+>", " ", raw_html)
                text_clean = _clean_extract(text_clean)
                section_content = f"== {matched_section.title} ==\n{text_clean}"
                
                logger.info(
                    "[NAVIGATOR] Targeted section retrieved: page=%r section=%r chars=%d",
                    article.page_title, matched_section.title, len(text_clean),
                )
                return RetrievalResult(
                    status=RetrievalStatus.SUCCESS,
                    query=f"{page_title} - {section_title}",
                    page_title=article.page_title,
                    page_url=f"{article.canonical_url}#{urllib.parse.quote(matched_section.title.replace(' ', '_'))}",
                    content=section_content,
                    metadata={
                        "strategy": "targeted_section_retrieval",
                        "section_title": matched_section.title,
                        "section_index": matched_section.index,
                        "canonical_title": article.page_title,
                    },
                )

            return RetrievalResult(
                status=RetrievalStatus.FAILED,
                query=f"{page_title} - {section_title}",
                metadata={"error": "Section fetch failed"},
            )

        finally:
            if own_client:
                await client.aclose()

    def discover_relevant_links(
        self,
        article: WikipediaArticle,
        query: str,
        active_need: Optional[Any] = None,
        max_links: int = 3,
    ) -> List[str]:
        """
        Filter internal Wikipedia links on an article that are directly relevant
        to the query or active InformationNeed. Rejects irrelevant links.
        """
        query_terms = set(_normalize(query).split())
        if active_need and getattr(active_need, "question", None):
            query_terms.update(_normalize(active_need.question).split())
        if active_need and getattr(active_need, "required_entities", None):
            for ent in active_need.required_entities:
                query_terms.update(_normalize(ent).split())

        stop = {"the", "and", "for", "with", "what", "why", "how", "who", "when", "where", "about", "his", "her", "their"}
        meaningful_terms = {t for t in query_terms if len(t) >= 3 and t not in stop}

        scored_links: List[Tuple[int, str]] = []
        for link in article.links:
            norm_link = _normalize(link)
            link_terms = set(norm_link.split())
            overlap = meaningful_terms & link_terms
            
            # Exact entity or substring match
            score = 0
            if any(norm_link == _normalize(term) for term in meaningful_terms):
                score += 50
            elif any(term in norm_link for term in meaningful_terms if len(term) >= 4):
                score += 25
            elif overlap:
                score += len(overlap) * 10

            if score > 0:
                scored_links.append((score, link))

        scored_links.sort(key=lambda x: -x[0])
        return [link for _, link in scored_links[:max_links]]

    def resolve_disambiguation(
        self,
        title: str,
        candidates: List[str],
        context_text: str,
        qu_intent: Optional[str] = None,
    ) -> Tuple[Optional[str], bool]:
        """
        Context-aware disambiguation selection.
        Returns (selected_candidate, requires_clarification).
        """
        if not candidates:
            return None, True

        norm_context = _normalize(context_text)
        scored: List[Tuple[int, str]] = []
        _DOMAIN_CUES = {
            "element": {"element", "atomic", "chemical", "chemistry", "metal", "liquid", "state", "periodic", "density", "toxicity"},
            "planet": {"planet", "solar", "orbit", "space", "sun", "astronomy", "celestial", "star", "distance"},
            "mythology": {"mythology", "god", "roman", "greek", "myth", "deity"},
            "automobile": {"automobile", "car", "vehicle", "ford", "motor", "drive"},
        }

        stop_cand = {"the", "and", "for", "with", "disambiguation"}
        for cand in candidates:
            norm_cand = _normalize(cand)
            cand_tokens = {w for w in re.findall(r"\b[a-z0-9]{3,}\b", norm_cand) if w not in stop_cand}
            
            # Direct token overlap with context
            overlap = [w for w in cand_tokens if w != _normalize(title) and w in norm_context]
            score = len(overlap) * 20

            # Qualifier in parentheses (e.g. "(element)" -> "element")
            m_qual = re.search(r"\((.*?)\)", norm_cand)
            if m_qual:
                qualifier = _normalize(m_qual.group(1))
                if qualifier in norm_context:
                    score += 50
                # Check domain semantic cues
                for domain_key, cue_words in _DOMAIN_CUES.items():
                    if domain_key in qualifier:
                        matched_cues = cue_words & set(re.findall(r"\b[a-z0-9]{3,}\b", norm_context))
                        score += len(matched_cues) * 15

            scored.append((score, cand))

        scored.sort(key=lambda x: -x[0])
        best_score, best_cand = scored[0]

        if best_score >= 15:
            logger.info("[NAVIGATOR] Disambiguation resolved: %r -> %r (score=%d)", title, best_cand, best_score)
            return best_cand, False

        logger.info("[NAVIGATOR] Disambiguation ambiguous for %r (top score=%d). Clarification required.", title, best_score)
        return None, True



wikipedia_navigator = WikipediaNavigator()
