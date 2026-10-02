import logging
import re
import time
import unicodedata
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from app.optimization.cache import local_cache_manager
from app.optimization.telemetry import telemetry_tracker
from app.persistence.article_store import wikipedia_article_store

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


class ArticleRelevanceResult(BaseModel):
    """Structured evaluation of article candidate relevance against an InformationNeed."""
    canonical_title: str
    entity_match: bool = False
    topic_match: bool = False
    relation_match: bool = False
    scope_match: bool = True
    temporal_match: bool = True
    relevance_score: float = 0.0
    accepted: bool = False
    is_navigation_candidate: bool = False
    reason: str = ""


# ── Normalization & token utilities ───────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


def _clean_extract(text: str) -> str:
    """Remove Wikipedia markup artifacts from REST API extracts."""
    text = re.sub(r"\[(?:O\.S\.|N\.S\.|Old\s+Style|New\s+Style)[^\]]*\]", "", text)
    text = re.sub(
        r"\[(?:citation\s+needed|clarification\s+needed|nb\s+\d+|note\s+\d+)[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\[\d{1,3}\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _title_case(text: str) -> str:
    """Convert 'narendra damodardas modi' -> 'Narendra Damodardas Modi'."""
    return " ".join(w.capitalize() for w in text.split())


_STOP_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "is", "was",
    "are", "were", "who", "what", "when", "where", "how", "list", "out",
    "all", "names", "tell", "give", "describe", "about", "and", "or", "with",
}


def _word_set(text: str) -> set:
    """Return set of meaningful words (3+ chars) from a normalized string, handling hyphens."""
    clean = re.sub(r"[-_/]+", " ", text)
    return {w for w in _normalize(clean).split() if len(w) >= 3 and w not in _STOP_WORDS}


# ── Distinct entity classification sets for contradiction detection ───────────

_COUNTRIES = {
    "india", "pakistan", "bangladesh", "sri lanka", "nepal", "bhutan", "china",
    "afghanistan", "iran", "iraq", "japan", "united states", "usa", "america",
    "united kingdom", "uk", "britain", "england", "france", "germany", "russia",
    "italy", "spain", "canada", "australia", "egypt", "turkey", "israel",
}

_DYNASTIES_EMPIRES = {
    "hoysala", "kushan", "mughal", "maurya", "gupta", "chola", "vijayanagara",
    "maratha", "ottoman", "roman", "byzantine", "abbasid", "wadiyar", "wodeyar",
    "chalukya", "rashtrakuta", "pallava", "satavahana", "seleucid",
}


# ── Candidate Relevance Gate ──────────────────────────────────────────────────

def evaluate_candidate_relevance(
    title: str,
    content: str,
    query: str,
    entity: Optional[str] = None,
    attribute: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    intent: Optional[str] = None,
) -> ArticleRelevanceResult:
    """
    Evaluate candidate article relevance against the structured InformationNeed.
    Deterministic gate that rejects entity mismatches (e.g. Pakistan for India, Kushan for Hoysala)
    and enforces LIST and temporal/scope contracts.
    """
    if not title:
        return ArticleRelevanceResult(
            canonical_title=title,
            accepted=False,
            reason="empty title",
        )

    norm_title = _normalize(title)
    norm_content = _normalize(content[:3000]) if content else ""
    norm_query = _normalize(query)
    norm_ent = _normalize(entity) if entity else ""
    constraints = constraints or {}

    temporal_str = constraints.get("temporal") or ""
    scope_str = constraints.get("scope") or ""
    if not temporal_str:
        m_t = re.search(r"\b(after|before|during|in)\s+([A-Za-z0-9\s]+)", query, re.I)
        if m_t:
            temporal_str = m_t.group(0).strip()
    if not scope_str:
        if "independent india" in norm_query or "independent" in norm_query:
            scope_str = "independent"

    # 1. Geopolitical / Country Contradiction Check
    target_countries = {c for c in _COUNTRIES if c in norm_ent or re.search(r"\b" + re.escape(c) + r"\b", norm_query)}
    title_countries = {c for c in _COUNTRIES if re.search(r"\b" + re.escape(c) + r"\b", norm_title)}
    if target_countries and title_countries:
        if not (target_countries & title_countries):
            req_c = list(target_countries)[0].title()
            cand_c = list(title_countries)[0].title()
            return ArticleRelevanceResult(
                canonical_title=title,
                entity_match=False,
                topic_match=False,
                relation_match=bool("prime minister" in norm_title or "ruler" in norm_title),
                scope_match=False,
                temporal_match=True,
                relevance_score=0.0,
                accepted=False,
                reason=f"country mismatch: requested {req_c} but candidate is {cand_c}",
            )

    # 2. Empire / Dynasty Contradiction Check
    target_dynasties = {d for d in _DYNASTIES_EMPIRES if d in norm_ent or re.search(r"\b" + re.escape(d) + r"\b", norm_query)}
    title_dynasties = {d for d in _DYNASTIES_EMPIRES if re.search(r"\b" + re.escape(d) + r"\b", norm_title)}
    if target_dynasties and title_dynasties:
        if not (target_dynasties & title_dynasties):
            req_d = list(target_dynasties)[0].title()
            cand_d = list(title_dynasties)[0].title()
            return ArticleRelevanceResult(
                canonical_title=title,
                entity_match=False,
                topic_match=False,
                relation_match=False,
                scope_match=False,
                temporal_match=True,
                relevance_score=0.0,
                accepted=False,
                reason=f"empire/dynasty mismatch: requested {req_d} but candidate is {cand_d}",
            )

    # 3. Entity Match
    entity_match = False
    title_stripped = re.sub(r"\s*\(.*?\)\s*$", "", norm_title).strip()

    if norm_ent:
        ent_words = _word_set(norm_ent)
        title_words = _word_set(norm_title)

        if norm_ent in _COUNTRIES:
            # If the requested entity is a country, the candidate article title MUST explicitly contain the country
            if re.search(r"\b" + re.escape(norm_ent) + r"\b", norm_title) or (ent_words & title_words):
                entity_match = True
            else:
                entity_match = False
        else:
            if norm_ent == norm_title or norm_ent == title_stripped:
                entity_match = True
            elif re.search(r"\b" + re.escape(norm_ent) + r"\b", norm_title):
                entity_match = True
            elif re.search(r"\b" + re.escape(norm_title) + r"\b", norm_ent):
                entity_match = True
            elif ent_words and (ent_words == title_words or ent_words.issubset(title_words)):
                entity_match = True
            elif ent_words and (len(ent_words & title_words) >= max(1, len(ent_words) - 1)):
                entity_match = True
    else:
        q_words = _word_set(query)
        t_words = _word_set(title)
        if q_words and (q_words & t_words):
            entity_match = True
        elif q_words and (_word_set(norm_content[:500]) & q_words):
            entity_match = True

    # 4. Relation / Attribute Match & LIST Contract
    relation_match = True
    is_nav_candidate = False
    relevance_reason = ""

    # If the candidate title is an exact match for the requested entity, it represents the subject itself
    is_exact_entity_article = bool(norm_ent and (norm_ent == norm_title or norm_ent == title_stripped))

    is_list_query = (
        (intent and str(intent).lower() in ("list", "intent.list"))
        or (attribute and attribute.lower() in ("rulers", "prime ministers", "presidents", "kings", "monarchs", "emperors", "dynasties", "list"))
        or any(w in norm_query for w in ("list out", "list the", "list all", "prime ministers of", "rulers of", "kings of"))
    )

    if is_list_query:
        # A LIST request requires a list/succession article or kingdom overview, not a single individual biography
        is_single_person_biography = False
        if any(w in norm_title for w in ("tipu sultan", "jawaharlal nehru", "narendra modi", "ashoka", "akbar")):
            if norm_ent and not any(w in norm_ent for w in ("sultan", "nehru", "modi", "ashoka", "akbar")):
                is_single_person_biography = True

        if is_single_person_biography:
            relation_match = False
            is_nav_candidate = True
            relevance_reason = "LIST requirement not satisfied: single ruler biography is navigation candidate, not ruler-list evidence"
        else:
            # Does candidate have evidence for the specific attribute?
            if attribute and attribute.lower() == "prime ministers":
                has_pm = "prime minister" in norm_title or "prime minister" in norm_content
                if not has_pm:
                    relation_match = False
                    relevance_reason = "candidate lacks prime minister evidence"
            elif attribute and attribute.lower() in ("rulers", "kings", "monarchs", "emperors"):
                has_rulers = (
                    any(w in norm_title for w in ("list", "ruler", "king", "dynasty", "monarch", "line of succession"))
                    or any(w in norm_content for w in ("succeeded by", "preceded by", "reign", "monarchs", "rulers", "line of rulers"))
                    or any(w in norm_title for w in ("kingdom", "empire"))
                )
                if not has_rulers:
                    relation_match = False
                    relevance_reason = "candidate lacks ruler/succession evidence"
            else:
                has_list_in_title = any(w in norm_title for w in ("list", "prime minister", "ruler", "king", "dynasty", "monarch", "line of succession"))
                has_list_in_content = any(w in norm_content for w in ("succeeded by", "preceded by", "reign", "monarchs", "prime ministers", "rulers", "line of rulers"))
                if not has_list_in_title and not has_list_in_content and not any(w in norm_title for w in ("kingdom", "empire")):
                    relation_match = False
                    relevance_reason = "candidate lacks list or succession evidence"
    elif attribute and not is_exact_entity_article:
        attr_lower = attribute.lower()
        if attr_lower in ("father", "mother", "parent", "son", "daughter"):
            if not any(w in norm_content for w in ("father", "mother", "parent", "born to", "son of", "daughter of")):
                relation_match = False
                relevance_reason = f"content lacks kinship attribute '{attribute}'"
        elif attr_lower in ("author", "founder", "creator"):
            if not any(w in norm_content for w in ("author", "wrote", "written by", "founder", "founded", "created")):
                relation_match = False
                relevance_reason = f"content lacks creator attribute '{attribute}'"

    # 5. Temporal Constraint Match
    temporal_match = True
    if temporal_str:
        norm_temp = _normalize(temporal_str)
        m_after = re.search(r"\bafter\s+([A-Za-z0-9\s]+)", norm_temp)
        if m_after:
            after_target = m_after.group(1).strip()
            # If candidate is the subject of "after", it cannot be final evidence for what came AFTER itself
            if after_target and (after_target in norm_title or norm_title in after_target):
                temporal_match = False
                is_nav_candidate = True
                relevance_reason = f"candidate is subject of temporal constraint '{temporal_str}', cannot be final evidence for rulers after subject"

    # 6. Scope Match
    scope_match = True
    if scope_str:
        norm_scope = _normalize(scope_str)
        if "independent" in norm_scope and "pakistan" in norm_title:
            scope_match = False
            relevance_reason = "country mismatch"

    accepted = bool(entity_match and relation_match and scope_match and temporal_match)
    if not accepted and not relevance_reason:
        if not entity_match:
            relevance_reason = "entity mismatch"
        elif not relation_match:
            relevance_reason = "relation mismatch"
        elif not temporal_match:
            relevance_reason = "temporal constraint not satisfied"
        elif not scope_match:
            relevance_reason = "scope mismatch"

    score = 0.0
    if entity_match:
        score += 50.0
    if relation_match:
        score += 30.0
    if scope_match:
        score += 10.0
    if temporal_match:
        score += 10.0

    return ArticleRelevanceResult(
        canonical_title=title,
        entity_match=entity_match,
        topic_match=entity_match or bool(query and _normalize(title) in norm_query),
        relation_match=relation_match,
        scope_match=scope_match,
        temporal_match=temporal_match,
        relevance_score=score,
        accepted=accepted,
        is_navigation_candidate=is_nav_candidate,
        reason=relevance_reason or "Relevance gate passed",
    )


def is_evidence_relevant(
    title: str,
    content: str,
    query: str,
    entity: Optional[str] = None,
    attribute: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    intent: Optional[str] = None,
) -> bool:
    """Evaluate if retrieved Wikipedia article is genuinely relevant to the query/entity."""
    gate = evaluate_candidate_relevance(
        title=title,
        content=content,
        query=query,
        entity=entity,
        attribute=attribute,
        constraints=constraints,
        intent=intent,
    )
    return gate.accepted


def _score_candidate(candidate_title: str, entity: Optional[str], query: str) -> int:
    """Score a Wikipedia search candidate against the target entity / query."""
    norm_title = _normalize(candidate_title)

    if entity:
        norm_entity = _normalize(entity)
        if norm_title == norm_entity:
            return 200
        title_stripped = re.sub(r"\s*\(.*?\)\s*$", "", norm_title).strip()
        if title_stripped == norm_entity:
            return 180
        if norm_entity in norm_title and norm_title in norm_entity:
            return 160
        if norm_title.startswith(norm_entity):
            return 140
        if norm_entity.startswith(norm_title):
            return 120

        title_words = _word_set(candidate_title)
        entity_words = _word_set(entity)
        if title_words and entity_words:
            overlap = title_words & entity_words
            ratio = len(overlap) / len(title_words)
            if ratio == 1.0:
                return 80
            elif ratio > 0.5:
                return 40

        try:
            if re.search(r"\b" + re.escape(norm_entity) + r"\b", norm_title):
                return 10
        except re.error:
            if norm_entity in norm_title:
                return 10

    query_words = _word_set(query)
    title_words = _word_set(candidate_title)
    if query_words and title_words:
        overlap = query_words & title_words
        if overlap:
            return int(30 * (len(overlap) / max(len(title_words), 1)))

    return 0


def _select_best_candidate(
    search_results: List[Dict],
    entity: Optional[str],
    query: str,
    attribute: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    intent: Optional[str] = None,
) -> str:
    """Pick the best Wikipedia candidate using relevance gating and entity-aware ranking."""
    if not search_results:
        raise ValueError("Empty search results")

    scored: List[Tuple[int, int, str]] = []  # (score, index, title)
    for idx, result in enumerate(search_results):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        if not title:
            continue

        gate = evaluate_candidate_relevance(
            title=title,
            content=snippet,
            query=query,
            entity=entity,
            attribute=attribute,
            constraints=constraints,
            intent=intent,
        )

        if not gate.accepted:
            logger.info(
                "[WIKI SEARCH] candidate=%r entity_match=%s scope_match=%s relation_match=%s temporal_match=%s accepted=false reason=%r",
                title,
                str(gate.entity_match).lower(),
                str(gate.scope_match).lower(),
                str(gate.relation_match).lower(),
                str(gate.temporal_match).lower(),
                gate.reason,
            )
            continue

        base_score = _score_candidate(title, entity, query)
        rank_bonus = max(0, 5 - idx)
        if re.search(r"\(disambiguation\)", title, re.IGNORECASE):
            base_score = max(0, base_score - 100)
        scored.append((base_score + rank_bonus, idx, title))

    if not scored:
        logger.warning(
            "[WIKI] All search candidates failed relevance gating for query=%r, entity=%r: %s",
            query, entity, [r.get("title") for r in search_results],
        )
        return ""

    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, best_idx, best_title = scored[0]
    logger.debug("[WIKI] Selected accepted candidate: %r (score=%d)", best_title, best_score)
    return best_title


# ── Retriever ─────────────────────────────────────────────────────────────────

def _parse_wikitables_to_text(wikitext: str) -> str:
    """
    Convert MediaWiki wikitables ({| ... |}) into clean plaintext structured records.
    Extracts table headers and rows as "Row Item (reign/details: ...)" lines.
    """
    if not wikitext or "{|" not in wikitext:
        return ""

    results = []
    # Find all tables in the text
    table_matches = re.findall(r"\{\|.*?\n\|\}(?=\n|$)", wikitext, re.DOTALL)
    for tbl in table_matches:
        raw_rows = tbl.split("|-")
        table_rows = []
        for r in raw_rows[1:]:
            lines = [line.strip() for line in r.split("\n") if line.strip() and line.strip().startswith("|")]
            cells = []
            for line in lines:
                content = line[1:].strip()
                subcells = content.split("||")
                for sc in subcells:
                    # extract link [[X|Y]] -> Y or [[X]] -> X
                    sc_clean = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", sc)
                    sc_clean = re.sub(r"<[^>]+>", "", sc_clean).strip()
                    # Strip citation refs like [1], [2]
                    sc_clean = re.sub(r"\[\d+\]", "", sc_clean).strip()
                    if sc_clean and not sc_clean.startswith("style=") and not sc_clean.startswith("class="):
                        cells.append(sc_clean)
            if len(cells) >= 2:
                if cells[0].isdigit() and len(cells) >= 3:
                    table_rows.append(f"{cells[1]} (reign: {cells[2]})")
                else:
                    table_rows.append(" - ".join(cells[:3]))
        if table_rows:
            results.append("\n".join(table_rows))

    return "\n\n".join(results)


def _split_extract_into_sections(canonical_title: str, extract: str) -> List[Any]:
    """
    Split a MediaWiki explaintext extract by section headers (== Section ==, === Sub ===).
    Returns a list of WikipediaDocumentSection instances.
    """
    from app.knowledge.document_models import WikipediaDocumentSection
    pattern = r"(^|\n)(={2,5})\s*([^=\n]+?)\s*\2\s*(?=\n|$)"
    matches = list(re.finditer(pattern, extract))

    sections: List[WikipediaDocumentSection] = []
    if not matches:
        if extract.strip():
            sections.append(
                WikipediaDocumentSection.create(
                    canonical_title=canonical_title,
                    title="Lead",
                    level=1,
                    index=0,
                    content=_clean_extract(extract),
                )
            )
        return sections

    # Lead section
    lead_content = extract[:matches[0].start()].strip()
    if lead_content:
        sections.append(
            WikipediaDocumentSection.create(
                canonical_title=canonical_title,
                title="Lead",
                level=1,
                index=0,
                content=_clean_extract(lead_content),
            )
        )

    current_parent_by_level: Dict[int, str] = {1: "Lead"}
    for i, m in enumerate(matches):
        level = len(m.group(2))
        title = m.group(3).strip()
        # Filter metadata sections
        if title.lower() in ("see also", "references", "external links", "further reading", "notes", "bibliography"):
            continue

        start_idx = m.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(extract)
        content = _clean_extract(extract[start_idx:end_idx].strip())

        # Determine parent
        parent = None
        for l in range(level - 1, 0, -1):
            if l in current_parent_by_level:
                parent = current_parent_by_level[l]
                break
        current_parent_by_level[level] = title

        sections.append(
            WikipediaDocumentSection.create(
                canonical_title=canonical_title,
                title=title,
                level=level,
                index=len(sections),
                content=content,
                parent_section_title=parent,
            )
        )

    return sections


def score_and_select_passages(
    passages: List[Any],
    query: str,
    entity: Optional[str] = None,
    attribute: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    max_passages: int = 4,
    max_total_chars: int = 2500,
) -> List[Any]:
    """
    Score WikipediaPassages against the InformationNeed (attribute, query, constraints).
    Returns top ranked passages bounded by token/character budget.
    """
    if not passages:
        return []

    norm_q = _normalize(query)
    q_words = {w for w in re.findall(r"\w+", norm_q) if len(w) >= 3 and w not in _STOP_WORDS}
    norm_attr = _normalize(attribute) if attribute else ""
    attr_words = {w for w in re.findall(r"\w+", norm_attr) if len(w) >= 3 and w not in _STOP_WORDS}
    norm_ent = _normalize(entity) if entity else ""

    # Temporal cues
    constraints = constraints or {}
    temporal_str = _normalize(str(constraints.get("temporal") or ""))
    temp_cues = set(re.findall(r"\w+", temporal_str)) - _STOP_WORDS

    # Relevant domain cues for common attributes
    attribute_cues: Dict[str, set[str]] = {
        "rulers": {"ruler", "rulers", "king", "kings", "monarch", "monarchs", "sultan", "sultans", "reign", "succession", "emperor", "dynasty", "throne", "wodeyar", "prince", "ascended"},
        "prime ministers": {"prime minister", "premier", "tenure", "cabinet", "elected", "office", "term"},
        "parents": {"father", "mother", "parent", "parents", "born to", "son of", "daughter of", "dasharatha", "kaushalya"},
        "founder": {"founder", "founded", "established", "created", "origin", "started", "co-founder"},
    }
    extra_cues = set()
    for k, cues in attribute_cues.items():
        if k in norm_attr or any(w in norm_attr for w in k.split()):
            extra_cues.update(cues)

    # Special Case 1: LIST rulers / dynasties query
    # If the document has specific dynasty / list of rulers sections (e.g. Vijayanagara: Sangama, Saluva, Tuluva, Aravidu),
    # aggregate them together so the entire chronological dynasty sequence is provided to Gemma.
    if norm_attr in ("rulers", "kings", "monarchs", "emperors", "dynasties") or "list" in norm_q:
        # Check if query also asks for battles or contributions
        has_battles = any(w in norm_q or w in norm_attr for w in ("battle", "battles", "war", "wars", "campaign"))
        has_contrib = any(w in norm_q or w in norm_attr for w in ("contribution", "contributions", "achievement", "administration", "reform"))
        
        if (has_battles or has_contrib) and not temporal_str:
            # Multi-facet gathering: rulers + battles + contributions
            facet_selected = []
            curr_chars = 0
            
            # 1. Lead / Overview
            lead_p = [p for p in passages if _normalize(p.section_title) in ("lead", "history", "overview")]
            if lead_p:
                p0 = lead_p[0]
                p0.score = 160.0
                facet_selected.append(p0)
                curr_chars += len(p0.text) + len(p0.section_title) + 10
                
            # 2. Dynasty / Rulers / Chhatrapatis / Peshwas
            ruler_p = [
                p for p in passages
                if any(k in _normalize(p.section_title) or k in _normalize(p.text[:150]) for k in ("dynasty", "rulers", "kings", "chhatrapati", "peshwa", "succession", "reign", "monarchs"))
                and p not in facet_selected
            ]
            for p in ruler_p[:3]:
                cost = len(p.text) + len(p.section_title) + 10
                if curr_chars + cost <= max_total_chars:
                    p.score = 150.0
                    facet_selected.append(p)
                    curr_chars += cost
                    
            # 3. Battles / Wars / Campaigns
            if has_battles:
                battle_p = [
                    p for p in passages
                    if any(k in _normalize(p.section_title) or k in _normalize(p.text[:150]) for k in ("battle", "war", "military", "campaign", "conflict", "panipat", "anglo-maratha", "conquest"))
                    and p not in facet_selected
                ]
                for p in battle_p[:2]:
                    cost = len(p.text) + len(p.section_title) + 10
                    if curr_chars + cost <= max_total_chars:
                        p.score = 140.0
                        facet_selected.append(p)
                        curr_chars += cost
                        
            # 4. Administration / Navy / Contributions
            if has_contrib:
                contrib_p = [
                    p for p in passages
                    if any(k in _normalize(p.section_title) or k in _normalize(p.text[:150]) for k in ("administration", "governance", "reforms", "navy", "forts", "economy", "legacy", "achievements"))
                    and p not in facet_selected
                ]
                for p in contrib_p[:2]:
                    cost = len(p.text) + len(p.section_title) + 10
                    if curr_chars + cost <= max_total_chars:
                        p.score = 135.0
                        facet_selected.append(p)
                        curr_chars += cost
                        
            if len(facet_selected) >= 2:
                return facet_selected

        dynasty_passages = [
            p for p in passages
            if any(k in _normalize(p.section_title) for k in ("dynasty", "rulers", "kings", "monarchs", "list of"))
            and p.text.strip()
        ]
        if len(dynasty_passages) >= 2 and not temporal_str:
            # Group dynasty passages chronologically by section_index
            dynasty_passages.sort(key=lambda x: (x.section_index, x.passage_index))
            selected_dynasty = []
            curr_chars = 0
            for p in dynasty_passages:
                cost = len(p.text) + len(p.section_title) + 10
                if curr_chars + cost > max_total_chars and selected_dynasty:
                    break
                p.score = 150.0
                selected_dynasty.append(p)
                curr_chars += cost
                if len(selected_dynasty) >= 8:
                    break
            if selected_dynasty:
                return selected_dynasty

    # Special Case 2: Conceptual Distinction ("why empire and not dynasty", "why is X called A rather than B")
    if norm_attr == "conceptual_distinction" or (
        "why" in norm_q and any(w in norm_q for w in ("not", "rather than", "instead of", "difference"))
    ):
        # We need both: 1) Lead/History/Name explaining the political empire/realm, and 2) Dynasty/Succession sections
        dist_selected = []
        curr_chars = 0

        # Pick lead or name/history passage first
        lead_or_name = [
            p for p in passages
            if _normalize(p.section_title) in ("lead", "name", "history", "background and origin theories")
        ]
        if lead_or_name:
            p0 = lead_or_name[0]
            p0.score = 150.0
            dist_selected.append(p0)
            curr_chars += len(p0.text) + len(p0.section_title) + 10

        # Pick dynasty / administration / governance passages
        dynasty_or_gov = [
            p for p in passages
            if any(k in _normalize(p.section_title) for k in ("dynasty", "governance", "administration", "early years", "defeat and decline"))
            and p not in dist_selected
        ]
        for p in dynasty_or_gov:
            cost = len(p.text) + len(p.section_title) + 10
            if curr_chars + cost > max_total_chars:
                break
            p.score = 120.0
            dist_selected.append(p)
            curr_chars += cost
            if len(dist_selected) >= 5:
                break

        if len(dist_selected) >= 2:
            return dist_selected

    scored: List[Tuple[float, Any]] = []
    for p in passages:
        score = 0.0
        p_text_norm = _normalize(p.text)
        s_title_norm = _normalize(p.section_title)

        # Attribute overlap in section title is huge
        if norm_attr and norm_attr in s_title_norm:
            score += 100.0
        elif attr_words:
            matched_sec_attr = [w for w in attr_words if w in s_title_norm]
            score += len(matched_sec_attr) * 40.0

        # Domain cues in section title or text
        for cue in extra_cues:
            if cue in s_title_norm:
                score += 50.0
            if cue in p_text_norm:
                score += 15.0

        # Temporal cues: "after X" should penalize sections explicitly titled "Under X" or "During X"
        # and boost sections describing succession, post-era, or "Princely state"
        if "after" in temporal_str:
            target_after = re.sub(r"^after\s+", "", temporal_str).strip()
            if target_after and (target_after in s_title_norm or any(w in s_title_norm for w in target_after.split() if w not in _STOP_WORDS)):
                if any(w in s_title_norm for w in ("under", "during", "reign of")):
                    score -= 100.0
            if "fall" in p_text_norm or "death" in p_text_norm or "following" in p_text_norm or "succeeded" in p_text_norm or "restoration" in s_title_norm or "princely" in s_title_norm:
                score += 80.0
        else:
            for tc in temp_cues:
                if tc in s_title_norm:
                    score += 40.0
                if tc in p_text_norm:
                    score += 25.0

        # General query word overlap
        overlap = [w for w in q_words if w in p_text_norm]
        score += len(overlap) * 10.0

        # Entity mention bonus (especially if not the main page title)
        if norm_ent and norm_ent in p_text_norm:
            score += 15.0

        # Lead section baseline
        if s_title_norm == "lead":
            score += 60.0

        p.score = score
        scored.append((score, p))

    scored.sort(key=lambda x: -x[0])

    selected: List[Any] = []
    curr_chars = 0
    for s, p in scored:
        if s <= 0.0 and len(selected) >= 1:
            break
        cost = len(p.text) + len(p.section_title) + 10
        if curr_chars + cost > max_total_chars and selected:
            break
        selected.append(p)
        curr_chars += cost
        if len(selected) >= max_passages:
            break

    # If nothing matched with high score, at least return the first passage (usually lead)
    if not selected and passages:
        selected = [passages[0]]

    return selected


# ── Retriever ─────────────────────────────────────────────────────────────────

class WikipediaRetriever:
    """Service responsible for querying Wikipedia APIs and retrieving verified article content."""

    SEARCH_API_URL = "https://en.wikipedia.org/w/api.php"
    USER_AGENT = "NexiaBot/1.0 (https://github.com/Deekshith1983/Nexia; contact: nexia-dev@example.com) httpx/0.27"

    async def _fetch_full_document(
        self, client: httpx.AsyncClient, title: str
    ) -> Optional[Any]:
        """
        Fetch full article wikitext/plain extracts via MediaWiki Action API (action=query&prop=extracts&explaintext=1).
        Constructs and returns a complete WikipediaDocument instance.
        """
        from app.knowledge.document_models import WikipediaDocument
        try:
            # 1. Action API call for clean full-text extract
            params = {
                "action": "query",
                "prop": "extracts|info",
                "inprop": "url",
                "exlimit": "1",
                "explaintext": "1",
                "titles": title,
                "redirects": 1,
                "format": "json",
            }
            resp = await client.get(self.SEARCH_API_URL, params=params)
            if resp.status_code != 200:
                return None

            pages = resp.json().get("query", {}).get("pages", {})
            if not pages:
                return None

            page_data = list(pages.values())[0]
            if "missing" in page_data:
                return None

            canonical_title = page_data.get("title", title)
            page_id = page_data.get("pageid")
            page_url = page_data.get("fullurl") or (
                summary_data.get("content_urls", {}).get("desktop", {}).get("page")
                if summary_data else f"https://en.wikipedia.org/wiki/{urllib.parse.quote(canonical_title.replace(' ', '_'))}"
            )
            full_extract = page_data.get("extract", "")
            if not full_extract:
                return None

            # Split into structured sections and passages
            sections = _split_extract_into_sections(canonical_title, full_extract)

            # Check if any key dynasty/ruler/list sections have empty content due to explaintext skipping wikitables
            empty_list_secs = [
                s for s in sections
                if not s.content.strip() and any(w in s.title.lower() for w in ("dynasty", "ruler", "king", "monarch", "list of"))
            ]
            if empty_list_secs:
                try:
                    # Fetch wikitext sections to parse wikitables
                    wt_params = {
                        "action": "parse",
                        "page": canonical_title,
                        "prop": "sections|wikitext",
                        "format": "json",
                    }
                    wt_resp = await client.get(self.SEARCH_API_URL, params=wt_params)
                    if wt_resp.status_code == 200:
                        wt_data = wt_resp.json().get("parse", {})
                        raw_wt = wt_data.get("wikitext", {}).get("*", "")
                        if raw_wt and "{|" in raw_wt:
                            # Parse sections from wikitext
                            wt_sections_split = re.split(r"(^|\n)(={2,5})\s*([^=\n]+?)\s*\2", raw_wt)
                            # Iterate and fill matching empty sections with parsed tables
                            for s in empty_list_secs:
                                s_title_norm = _normalize(s.title)
                                # Search in wikitext for this heading
                                sec_pattern = rf"={{{s.level}}}\s*{re.escape(s.title)}\s*={{{s.level}}}(.*?)(?=\n={{{s.level}}}|\Z)"
                                sec_match = re.search(sec_pattern, raw_wt, re.DOTALL | re.IGNORECASE)
                                if sec_match:
                                    sec_wt = sec_match.group(1)
                                    parsed_tbl = _parse_wikitables_to_text(sec_wt)
                                    if parsed_tbl:
                                        from app.knowledge.document_models import WikipediaDocumentSection
                                        # Recreate section with parsed table text
                                        updated_sec = WikipediaDocumentSection.create(
                                            canonical_title=canonical_title,
                                            title=s.title,
                                            level=s.level,
                                            index=s.index,
                                            content=parsed_tbl,
                                            parent_section_title=s.parent_section_title,
                                        )
                                        # Replace in sections list
                                        for idx, orig_sec in enumerate(sections):
                                            if orig_sec.section_id == s.section_id:
                                                sections[idx] = updated_sec
                                                break
                except Exception as wt_err:
                    logger.debug("[WIKI] wikitext table parsing error: %s", wt_err)

            # Lead summary
            lead_summary = ""
            if sections and sections[0].title == "Lead":
                lead_summary = sections[0].content

            doc = WikipediaDocument.create(
                canonical_title=canonical_title,
                url=page_url,
                lead_summary=lead_summary,
                page_id=page_id,
                sections=sections,
            )
            return doc
        except Exception as e:
            logger.warning("[WIKI] _fetch_full_document error for %r: %s", title, e)
            return None

    def _entity_title_variants(self, entity: str) -> List[str]:
        """Generate title variants for direct Wikipedia lookup."""
        raw = entity.strip()
        tc = _title_case(raw)
        variants: List[str] = []

        variants.append(tc)
        if raw != tc:
            variants.append(raw)

        # Historical kingdom/dynasty/empire inversions
        # e.g. "Mysore Kingdom" -> "Kingdom of Mysore"
        m_k = re.match(r"^(.+?)\s+(kingdom|empire|dynasty|sultanate)$", tc, re.I)
        if m_k:
            inv = f"{m_k.group(2).capitalize()} of {m_k.group(1)}"
            if inv not in variants:
                variants.append(inv)
            core = m_k.group(1)
            if core not in variants and len(core) >= 4:
                variants.append(core)

        m_of = re.match(r"^(kingdom|empire|dynasty|sultanate)\s+of\s+(.+)$", tc, re.I)
        if m_of:
            inv = f"{m_of.group(2)} {m_of.group(1).capitalize()}"
            if inv not in variants:
                variants.append(inv)

        # Inversion for "Jaya and Vijaya" -> "Jaya-Vijaya"
        if " and " in tc.lower():
            hyphenated = re.sub(r"\s+and\s+", "-", tc, flags=re.I)
            if hyphenated not in variants:
                variants.append(hyphenated)

        # First + last word variant
        words = raw.split()
        if len(words) >= 3 and not m_k and not m_of:
            short = f"{words[0].capitalize()} {words[-1].capitalize()}"
            if short not in variants:
                variants.append(short)

        return variants

    async def retrieve(
        self,
        query: str,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        force_refresh: bool = False,
        request_id: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve Wikipedia evidence with 3-level cascade & candidate relevance gate:
        1. In-memory / fast local cache
        2. Persistent SQLite WikipediaArticleStore (FTS5 indexed)
        3. Wikipedia API network requests
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RetrievalResult(status=RetrievalStatus.NO_RESULTS, query=cleaned_query)

        # Infer QU context if not explicitly passed
        if not attribute or not intent or not constraints:
            try:
                from app.core.query_understanding import query_analyzer
                qu_inferred = query_analyzer.understand(cleaned_query)
                entity = entity or (qu_inferred.entity if qu_inferred.entities else None)
                attribute = attribute or qu_inferred.attribute
                intent = intent or (qu_inferred.intent.value if hasattr(qu_inferred.intent, "value") else str(qu_inferred.intent))
                merged_constraints = dict(qu_inferred.constraints or {})
                if constraints:
                    merged_constraints.update(constraints)
                constraints = merged_constraints
            except Exception as qu_err:
                logger.debug("[WIKI] automatic QU inference in retriever: %s", qu_err)

        # ── Level 1: Check Memory / Fast Local Cache ──────────────────────────
        try:
            cached_result = local_cache_manager.get_wikipedia(cleaned_query, entity=entity, force_refresh=force_refresh)
            if cached_result is not None:
                gate = evaluate_candidate_relevance(
                    title=cached_result.page_title or "",
                    content=cached_result.content or "",
                    query=cleaned_query,
                    entity=entity,
                    attribute=attribute,
                    constraints=constraints,
                    intent=intent,
                )
                if gate.accepted:
                    try:
                        telemetry_tracker.record_wikipedia_call(request_id, duration_ms=0.5, cached=True)
                    except Exception as t_exc:
                        logger.debug("[WIKI] telemetry record cached call error: %s", t_exc)
                    # Create a copy with cached=True flag
                    cached_meta = dict(cached_result.metadata)
                    cached_meta["cached"] = True
                    return RetrievalResult(
                        status=cached_result.status,
                        query=cached_result.query,
                        page_title=cached_result.page_title,
                        page_url=cached_result.page_url,
                        content=cached_result.content,
                        source=cached_result.source,
                        metadata=cached_meta,
                    )
                else:
                    logger.info(
                        "[WIKI CACHE] candidate=%r entity_match=%s scope_match=%s relation_match=%s temporal_match=%s accepted=false reason=%r",
                        cached_result.page_title,
                        str(gate.entity_match).lower(),
                        str(gate.scope_match).lower(),
                        str(gate.relation_match).lower(),
                        str(gate.temporal_match).lower(),
                        gate.reason,
                    )
        except Exception as c_exc:
            logger.warning("[WIKI] local cache read failed for query=%r: %s", cleaned_query, c_exc)

        # ── Level 2: Check Persistent SQLite WikipediaArticleStore ────────────
        if not force_refresh:
            try:
                cached_doc = None
                # Step 1: Exact canonical title lookup for entity & variants
                if entity and entity.strip():
                    cached_doc = wikipedia_article_store.get_document(entity)
                    if not cached_doc:
                        for variant in self._entity_title_variants(entity):
                            cached_doc = wikipedia_article_store.get_document(variant)
                            if cached_doc:
                                break

                # Step 2: Exact canonical title lookup for query (if title-like)
                if not cached_doc and len(cleaned_query.split()) <= 4 and not any(w in cleaned_query.lower() for w in ("what", "who", "why", "list", "tell", "how", "when")):
                    cached_doc = wikipedia_article_store.get_document(cleaned_query)

                if cached_doc and (cached_doc.lead_summary or cached_doc.sections):
                    art_title = cached_doc.canonical_title
                    art_summary = cached_doc.lead_summary or (cached_doc.sections[0].content if cached_doc.sections else "")
                    gate = evaluate_candidate_relevance(
                        title=art_title,
                        content=art_summary,
                        query=cleaned_query,
                        entity=entity,
                        attribute=attribute,
                        constraints=constraints,
                        intent=intent,
                    )
                    if gate.accepted:
                        logger.info(
                            "[WIKI] Reused persistent document store entry for %r -> %r",
                            cleaned_query, art_title,
                        )
                        # Extract and score targeted passages
                        all_passages = cached_doc.get_passages()
                        chosen_passages = score_and_select_passages(
                            all_passages,
                            query=cleaned_query,
                            entity=entity,
                            attribute=attribute,
                            constraints=constraints,
                            max_passages=4,
                            max_total_chars=2500,
                        )
                        if chosen_passages:
                            content_body = "\n\n".join(
                                f"[{p.section_title}]\n{p.text}" for p in chosen_passages
                            )
                        else:
                            content_body = art_summary

                        result = RetrievalResult(
                            status=RetrievalStatus.SUCCESS,
                            query=cleaned_query,
                            page_title=art_title,
                            page_url=cached_doc.url,
                            content=content_body,
                            metadata={
                                "strategy": "persistent_document_store",
                                "cached": True,
                                "source": "ArticleStore",
                                "passage_count": len(chosen_passages),
                                "sections_used": list({p.section_title for p in chosen_passages}),
                            },
                        )
                        try:
                            telemetry_tracker.record_wikipedia_call(request_id, duration_ms=0.5, cached=True)
                        except Exception:
                            pass
                        try:
                            local_cache_manager.set_wikipedia(cleaned_query, result, entity=entity)
                        except Exception:
                            pass
                        return result
                    else:
                        logger.info(
                            "[WIKI CACHE] candidate=%r entity_match=%s scope_match=%s relation_match=%s temporal_match=%s accepted=false reason=%r",
                            art_title,
                            str(gate.entity_match).lower(),
                            str(gate.scope_match).lower(),
                            str(gate.relation_match).lower(),
                            str(gate.temporal_match).lower(),
                            gate.reason,
                        )

                # Step 3: FTS candidate discovery (Candidates MUST pass relevance gate!)
                fts_search_term = entity if (entity and entity.strip()) else " ".join([w for w in cleaned_query.split() if len(w) >= 3 and w.lower() not in _STOP_WORDS])
                if fts_search_term:
                    candidates = wikipedia_article_store.search_articles(fts_search_term, limit=5)
                    for cand in candidates:
                        cand_title = cand.get("canonical_title", cand.get("title", ""))
                        cand_summary = cand.get("summary") or cand.get("content", "")
                        gate = evaluate_candidate_relevance(
                            title=cand_title,
                            content=cand_summary,
                            query=cleaned_query,
                            entity=entity,
                            attribute=attribute,
                            constraints=constraints,
                            intent=intent,
                        )
                        if gate.accepted:
                            cand_doc = wikipedia_article_store.get_document(cand_title)
                            if cand_doc:
                                all_passages = cand_doc.get_passages()
                                chosen_passages = score_and_select_passages(
                                    all_passages,
                                    query=cleaned_query,
                                    entity=entity,
                                    attribute=attribute,
                                    constraints=constraints,
                                    max_passages=4,
                                    max_total_chars=2500,
                                )
                                content_body = "\n\n".join(
                                    f"[{p.section_title}]\n{p.text}" for p in chosen_passages
                                ) if chosen_passages else cand_summary
                            else:
                                content_body = cand_summary
                                chosen_passages = []

                            logger.info(
                                "[WIKI] Reused persistent document store FTS entry for %r -> %r",
                                cleaned_query, cand_title,
                            )
                            result = RetrievalResult(
                                status=RetrievalStatus.SUCCESS,
                                query=cleaned_query,
                                page_title=cand_title,
                                page_url=cand.get("url", ""),
                                content=content_body,
                                metadata={
                                    "strategy": "persistent_document_store_fts",
                                    "cached": True,
                                    "source": "ArticleStore",
                                    "passage_count": len(chosen_passages),
                                    "sections_used": list({p.section_title for p in chosen_passages}) if chosen_passages else [],
                                },
                            )
                            try:
                                telemetry_tracker.record_wikipedia_call(request_id, duration_ms=0.5, cached=True)
                            except Exception:
                                pass
                            try:
                                local_cache_manager.set_wikipedia(cleaned_query, result, entity=entity)
                            except Exception:
                                pass
                            return result
                        else:
                            logger.info(
                                "[WIKI CACHE] candidate=%r entity_match=%s scope_match=%s relation_match=%s temporal_match=%s accepted=false reason=%r",
                                cand_title,
                                str(gate.entity_match).lower(),
                                str(gate.scope_match).lower(),
                                str(gate.relation_match).lower(),
                                str(gate.temporal_match).lower(),
                                gate.reason,
                            )
            except Exception as store_err:
                logger.warning("[WIKI] article store lookup failed: %s", store_err)

        # ── Level 3: Wikipedia API Network Request ────────────────────────────
        t_start = time.monotonic()
        headers = {"User-Agent": self.USER_AGENT}
        logger.debug("[WIKI] retrieve (network) query=%r entity=%r attr=%r", cleaned_query, entity, attribute)

        try:
            async with httpx.AsyncClient(timeout=6.0, headers=headers) as client:

                # ── Strategy 1: Direct entity title lookup ────────────────────
                if entity and entity.strip():
                    for variant in self._entity_title_variants(entity):
                        doc = await self._fetch_full_document(client, variant)
                        if doc:
                            actual_title = doc.canonical_title
                            extract = doc.lead_summary or (doc.sections[0].content if doc.sections else "")
                            page_url = doc.url
                            gate = evaluate_candidate_relevance(
                                title=actual_title,
                                content=extract,
                                query=cleaned_query,
                                entity=entity,
                                attribute=attribute,
                                constraints=constraints,
                                intent=intent,
                            )
                            if not gate.accepted:
                                logger.info(
                                    "[WIKI SEARCH] candidate=%r entity_match=%s scope_match=%s relation_match=%s temporal_match=%s accepted=false reason=%r",
                                    actual_title,
                                    str(gate.entity_match).lower(),
                                    str(gate.scope_match).lower(),
                                    str(gate.relation_match).lower(),
                                    str(gate.temporal_match).lower(),
                                    gate.reason,
                                )
                                continue

                            all_passages = doc.get_passages()
                            chosen_passages = score_and_select_passages(
                                all_passages,
                                query=cleaned_query,
                                entity=entity,
                                attribute=attribute,
                                constraints=constraints,
                                max_passages=4,
                                max_total_chars=2500,
                            )
                            if chosen_passages:
                                content_body = "\n\n".join(
                                    f"[{p.section_title}]\n{p.text}" for p in chosen_passages
                                )
                            else:
                                content_body = extract
                            sections_used = list({p.section_title for p in chosen_passages})
                            passage_count = len(chosen_passages)

                            result = RetrievalResult(
                                status=RetrievalStatus.SUCCESS,
                                query=cleaned_query,
                                page_title=actual_title,
                                page_url=page_url,
                                content=content_body,
                                metadata={
                                    "strategy": "direct_entity_lookup",
                                    "lookup_variant": variant,
                                    "page_id": doc.page_id,
                                    "passage_count": passage_count,
                                    "sections_used": sections_used,
                                },
                            )
                            dur_ms = (time.monotonic() - t_start) * 1000.0
                            try:
                                telemetry_tracker.record_wikipedia_call(request_id, duration_ms=dur_ms, cached=False)
                            except Exception as t_exc:
                                logger.debug("[WIKI] telemetry record error: %s", t_exc)

                            try:
                                local_cache_manager.set_wikipedia(cleaned_query, result, entity=entity)
                            except Exception as c_exc:
                                logger.warning("[WIKI] cache write failed: %s", c_exc)

                            aliases_to_save = []
                            for al in [entity, variant]:
                                if al and _normalize(al) != _normalize(actual_title) and al not in aliases_to_save:
                                    aliases_to_save.append(al)

                            try:
                                doc.aliases = aliases_to_save
                                doc.metadata["entity"] = entity
                                doc.metadata["attribute"] = attribute
                                wikipedia_article_store.save_document(doc)
                            except Exception as s_exc:
                                logger.warning("[WIKI] article store save failed: %s", s_exc)

                            return result

                    logger.debug(
                        "[WIKI] all direct lookup variants failed for entity=%r; falling back to search",
                        entity,
                    )

                # ── Strategy 2: Structured Full-Text Search ───────────────────
                if entity and attribute:
                    if attribute.lower() in ("rulers", "kings", "monarchs", "emperors"):
                        search_target = f"List of rulers of {entity}"
                    elif attribute.lower() == "prime ministers":
                        search_target = f"Prime ministers of {entity}"
                    else:
                        search_target = f"{entity} {attribute}"
                elif entity:
                    search_target = entity
                else:
                    search_target = cleaned_query

                search_params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": search_target,
                    "srlimit": 5,
                    "utf8": 1,
                    "format": "json",
                }
                search_res = await client.get(self.SEARCH_API_URL, params=search_params)

                search_results = []
                if search_res.status_code == 200:
                    search_results = search_res.json().get("query", {}).get("search", [])

                # Fallback to query search if structured search returned no results
                if not search_results and search_target != cleaned_query:
                    search_params["srsearch"] = cleaned_query
                    fb_res = await client.get(self.SEARCH_API_URL, params=search_params)
                    if fb_res.status_code == 200:
                        search_results = fb_res.json().get("query", {}).get("search", [])

                if not search_results:
                    dur_ms = (time.monotonic() - t_start) * 1000.0
                    try:
                        telemetry_tracker.record_wikipedia_call(request_id, duration_ms=dur_ms, cached=False)
                    except Exception:
                        pass
                    return RetrievalResult(status=RetrievalStatus.NO_RESULTS, query=cleaned_query)

                selected_title = _select_best_candidate(
                    search_results,
                    entity=entity,
                    query=cleaned_query,
                    attribute=attribute,
                    constraints=constraints,
                    intent=intent,
                )

                if not selected_title:
                    dur_ms = (time.monotonic() - t_start) * 1000.0
                    try:
                        telemetry_tracker.record_wikipedia_call(request_id, duration_ms=dur_ms, cached=False)
                    except Exception:
                        pass
                    return RetrievalResult(status=RetrievalStatus.NO_RESULTS, query=cleaned_query)

                doc = await self._fetch_full_document(client, selected_title)
                if not doc:
                    dur_ms = (time.monotonic() - t_start) * 1000.0
                    try:
                        telemetry_tracker.record_wikipedia_call(request_id, duration_ms=dur_ms, cached=False)
                    except Exception:
                        pass
                    return RetrievalResult(
                        status=RetrievalStatus.NO_RESULTS,
                        query=cleaned_query,
                        page_title=selected_title,
                    )

                actual_title = doc.canonical_title
                extract = doc.lead_summary or (doc.sections[0].content if doc.sections else "")
                page_url = doc.url

                final_gate = evaluate_candidate_relevance(
                    title=actual_title,
                    content=extract,
                    query=cleaned_query,
                    entity=entity,
                    attribute=attribute,
                    constraints=constraints,
                    intent=intent,
                )

                if not final_gate.accepted:
                    logger.info(
                        "[WIKI SEARCH] candidate=%r entity_match=%s scope_match=%s relation_match=%s temporal_match=%s accepted=false reason=%r",
                        actual_title,
                        str(final_gate.entity_match).lower(),
                        str(final_gate.scope_match).lower(),
                        str(final_gate.relation_match).lower(),
                        str(final_gate.temporal_match).lower(),
                        final_gate.reason,
                    )
                    return RetrievalResult(
                        status=RetrievalStatus.NO_RESULTS,
                        query=cleaned_query,
                        page_title=actual_title,
                        metadata={"reason": "relevance_gate_failed", "rejected_title": actual_title, "gate_reason": final_gate.reason},
                    )

                # Fetch full WikipediaDocument via Action API
                all_passages = doc.get_passages()
                chosen_passages = score_and_select_passages(
                    all_passages,
                    query=cleaned_query,
                    entity=entity,
                    attribute=attribute,
                    constraints=constraints,
                    max_passages=4,
                    max_total_chars=2500,
                )
                if chosen_passages:
                    content_body = "\n\n".join(
                        f"[{p.section_title}]\n{p.text}" for p in chosen_passages
                    )
                else:
                    content_body = extract
                sections_used = list({p.section_title for p in chosen_passages})
                passage_count = len(chosen_passages)

                result = RetrievalResult(
                    status=RetrievalStatus.SUCCESS,
                    query=cleaned_query,
                    page_title=actual_title,
                    page_url=page_url,
                    content=content_body,
                    metadata={
                        "strategy": "search_ranked",
                        "snippet": search_results[0].get("snippet", ""),
                        "page_id": doc.page_id,
                        "candidates": [r.get("title") for r in search_results],
                        "passage_count": passage_count,
                        "sections_used": sections_used,
                    },
                )
                dur_ms = (time.monotonic() - t_start) * 1000.0
                try:
                    telemetry_tracker.record_wikipedia_call(request_id, duration_ms=dur_ms, cached=False)
                except Exception as t_exc:
                    logger.debug("[WIKI] telemetry record error: %s", t_exc)

                try:
                    local_cache_manager.set_wikipedia(cleaned_query, result, entity=entity)
                except Exception as c_exc:
                    logger.warning("[WIKI] cache write failed: %s", c_exc)

                aliases_to_save = []
                if entity and _normalize(entity) != _normalize(actual_title):
                    aliases_to_save.append(entity)

                try:
                    if doc:
                        doc.aliases = aliases_to_save
                        doc.metadata["entity"] = entity
                        doc.metadata["attribute"] = attribute
                        wikipedia_article_store.save_document(doc)
                    else:
                        wikipedia_article_store.save_article(
                            title=actual_title,
                            url=page_url,
                            summary=extract,
                            content=extract,
                            aliases=aliases_to_save,
                            metadata={"entity": entity, "attribute": attribute},
                        )
                except Exception as s_exc:
                    logger.warning("[WIKI] article store save failed: %s", s_exc)

                return result

        except (httpx.HTTPError, httpx.TimeoutException, Exception) as e:
            dur_ms = (time.monotonic() - t_start) * 1000.0
            try:
                telemetry_tracker.record_wikipedia_call(request_id, duration_ms=dur_ms, cached=False)
            except Exception:
                pass
            logger.warning("[WIKI] retrieval failed: %s", e)
            return RetrievalResult(
                status=RetrievalStatus.FAILED,
                query=cleaned_query,
                metadata={"error": str(e)},
            )


wikipedia_retriever = WikipediaRetriever()
