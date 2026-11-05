"""Single Primary Article Citation Manager for Nexia Answers.

Evaluates evidence passages and verified factual claims to select a single,
authoritative primary Wikipedia article reference, and formats the user-facing
answer with a clean trailing markdown hyperlink:
"[Article Title](canonical_article_url)"
without inline numbered citations, raw URLs, or separate bibliographies.
"""

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from app.retrieval.nlp_query_parser import NLPQueryParser

logger = logging.getLogger("nexia.citations")


@dataclass
class PrimaryArticle:
    """Selected primary Wikipedia article reference."""

    title: str
    url: str
    supported_claims: int = 0
    total_claims: int = 0
    coverage_score: float = 0.0
    retrieval_score: float = 0.0


class PrimaryArticleSelector:
    """Evaluates contributing articles against verified claims to select a single primary reference."""

    DEFAULT_COVERAGE_THRESHOLD: float = 0.35

    # Common English words that must not be counted as significant content words
    GENERIC_STOPWORDS: Set[str] = {
        "that", "this", "these", "those", "with", "from", "have", "were", "been",
        "their", "which", "about", "into", "more", "other", "some", "time", "when",
        "what", "where", "text", "also", "such", "than", "them", "then", "there",
        "each", "many", "most", "only", "well", "will", "would", "could", "should",
    }

    @classmethod
    def select_primary_article(
        cls,
        selected_chunks: List[Any],
        answer_text: str,
        question: str,
        subqueries: Optional[List[str]] = None,
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    ) -> Optional[PrimaryArticle]:
        """Select a single primary Wikipedia article with the strongest verified evidence coverage.

        Returns None if no single article provides adequate coverage (safe fallback).
        """
        if not selected_chunks or not answer_text.strip():
            return None

        # 1. Parse sentences and extract claim-to-chunk links
        paragraphs = answer_text.strip().split("\n\n")
        sentences: List[str] = []
        for p in paragraphs:
            # Skip pure list headers
            if p.strip().endswith(":") and len(p.strip().split("\n")) == 1:
                continue
            for s in re.split(r"(?<=[.!?])\s+", p.strip()):
                s_clean = s.strip()
                if s_clean:
                    sentences.append(s_clean)

        total_claims = len(sentences)
        if total_claims == 0:
            return None

        # Group chunks by canonical article identity (normalized canonical URL or title)
        article_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for idx, chunk in enumerate(selected_chunks, start=1):
            if isinstance(chunk, dict):
                doc_title = chunk.get("document_title") or chunk.get("title") or "Unknown Source"
                source_url = chunk.get("source_url") or chunk.get("url") or ""
                rrf_score = chunk.get("rrf_score", 0.0)
                text = chunk.get("passage_text") or chunk.get("text") or ""
            else:
                doc_title = getattr(chunk, "document_title", None) or getattr(chunk, "title", "Unknown Source")
                source_url = getattr(chunk, "source_url", None) or getattr(chunk, "url", "")
                rrf_score = getattr(chunk, "rrf_score", 0.0)
                text = getattr(chunk, "passage_text", "") or getattr(chunk, "text", "")

            if not source_url and doc_title != "Unknown Source":
                source_url = f"https://en.wikipedia.org/wiki/{doc_title.replace(' ', '_')}"

            canonical_id = cls._canonical_article_id(doc_title, source_url)
            if canonical_id not in article_chunks:
                article_chunks[canonical_id] = []
            article_chunks[canonical_id].append({
                "1_based_idx": idx,
                "title": doc_title,
                "url": source_url,
                "chunk": chunk,
                "rrf_score": rrf_score,
                "text": text,
            })

        if not article_chunks:
            return None

        # 2. Map sentences to supporting articles
        article_supported_claims: Dict[str, Set[int]] = {k: set() for k in article_chunks}
        chunk_to_article: Dict[int, str] = {}
        for art_id, c_list in article_chunks.items():
            for c_info in c_list:
                chunk_to_article[c_info["1_based_idx"]] = art_id

        for s_idx, sent in enumerate(sentences):
            cites = [int(m) for m in re.findall(r"\[(\d+)\]", sent)]
            matched_article = False
            for cite in cites:
                if cite in chunk_to_article:
                    article_supported_claims[chunk_to_article[cite]].add(s_idx)
                    matched_article = True

            # If no inline citation or citation was stripped/unmatched, verify strong lexical support
            if not matched_article:
                s_words = {
                    w.lower()
                    for w in re.findall(r"\b[a-zA-Z]{4,}\b", sent)
                    if w.lower() not in NLPQueryParser.STOPWORDS_CHUNKS and w.lower() not in cls.GENERIC_STOPWORDS
                }
                if len(s_words) >= 3:
                    best_art = None
                    max_matches = 0
                    for art_id, c_list in article_chunks.items():
                        for c_info in c_list:
                            c_text = c_info["text"].lower()
                            matches = sum(1 for w in s_words if w in c_text)
                            # Require at least 3 significant words and at least 30% of sentence vocabulary
                            if matches >= 3 and (matches / len(s_words)) >= 0.30 and matches > max_matches:
                                max_matches = matches
                                best_art = art_id
                    if best_art:
                        article_supported_claims[best_art].add(s_idx)

        # 3. Score each article using verified coverage, relevance, and query focus
        q_words = {
            w.lower()
            for w in re.findall(r"\b[a-zA-Z]{3,}\b", question)
            if w.lower() not in NLPQueryParser.STOPWORDS_CHUNKS and w.lower() not in cls.GENERIC_STOPWORDS
        }
        subquery_list = subqueries or [question]
        total_subqueries = max(len(subquery_list), 1)

        candidate_scores: List[PrimaryArticle] = []

        for art_id, c_list in article_chunks.items():
            first_entry = c_list[0]
            title = first_entry["title"]
            url = first_entry["url"]

            supported_count = len(article_supported_claims[art_id])
            claim_prop = supported_count / total_claims if total_claims > 0 else 0.0

            # Mean RRF / relevance score across contributing passages
            rrf_vals = [c["rrf_score"] for c in c_list if c["rrf_score"] > 0]
            avg_rrf = sum(rrf_vals) / len(rrf_vals) if rrf_vals else 0.0
            norm_rrf = min(avg_rrf / 0.50, 1.0) if avg_rrf > 0 else 0.0

            # Subquery coverage
            covered_sq = 0
            for sq in subquery_list:
                sq_words = {
                    w.lower()
                    for w in re.findall(r"\b[a-zA-Z]{3,}\b", sq)
                    if w.lower() not in NLPQueryParser.STOPWORDS_CHUNKS and w.lower() not in cls.GENERIC_STOPWORDS
                }
                if sq_words and any(any(w in c["text"].lower() for w in sq_words) for c in c_list):
                    covered_sq += 1
            sq_ratio = covered_sq / total_subqueries

            # Query title/entity match bonus
            t_words = {
                w.lower()
                for w in re.findall(r"\b[a-zA-Z]{3,}\b", title)
                if w.lower() not in cls.GENERIC_STOPWORDS
            }
            entity_bonus = 0.15 if (t_words & q_words) else 0.0

            # Composite coverage score: claim support dominates (50%), relevance (30%), subqueries (20%)
            composite_score = (0.50 * claim_prop) + (0.30 * norm_rrf) + (0.20 * sq_ratio) + entity_bonus

            candidate_scores.append(
                PrimaryArticle(
                    title=title,
                    url=url,
                    supported_claims=supported_count,
                    total_claims=total_claims,
                    coverage_score=round(composite_score, 4),
                    retrieval_score=round(avg_rrf, 4),
                )
            )

        # Sort candidates deterministically:
        candidate_scores.sort(
            key=lambda a: (a.coverage_score, a.supported_claims, a.retrieval_score, a.title),
            reverse=True,
        )

        best = candidate_scores[0]

        # Selection verification rules:
        # Rule 1: Must support at least 1 verified claim
        if best.supported_claims == 0:
            logger.info("Primary article selection fallback: No article supported verified claims.")
            return None

        # Rule 2: Claim proportion must be adequate (at least 35% of answer claims supported by this article)
        claim_ratio = best.supported_claims / total_claims if total_claims > 0 else 0.0
        if claim_ratio < 0.35:
            logger.info("Primary article selection fallback: Claim ratio %.2f < 0.35", claim_ratio)
            return None

        # Rule 3: Relevance must not be negligible
        if best.retrieval_score < 0.10:
            logger.info("Primary article selection fallback: Retrieval score %.2f < 0.10", best.retrieval_score)
            return None

        # Rule 4: Must meet the overall coverage threshold
        if best.coverage_score < coverage_threshold:
            logger.info(
                "Primary article selection fallback: Best candidate '%s' score %.2f < threshold %.2f",
                best.title,
                best.coverage_score,
                coverage_threshold,
            )
            return None

        # Rule 5: If multiple articles exist and top two are tied with disjoint partial coverage
        if len(candidate_scores) > 1:
            second = candidate_scores[1]
            if (
                best.supported_claims == second.supported_claims
                and claim_ratio < 0.40
                and abs(best.coverage_score - second.coverage_score) < 0.05
            ):
                logger.info(
                    "Primary article selection fallback: Ambiguous tie between '%s' and '%s' with low individual coverage.",
                    best.title,
                    second.title,
                )
                return None

        return best

    @staticmethod
    def _canonical_article_id(title: str, url: str) -> str:
        """Derive normalized canonical article identifier from title or URL."""
        if url:
            parsed = urlparse(url)
            path = parsed.path.rstrip("/")
            if "/wiki/" in path:
                return path.split("/wiki/")[-1].lower()
            if path:
                return path.lower()
        return title.strip().lower().replace(" ", "_")


def strip_inline_citations(text: str) -> str:
    """Remove user-facing numbered citations like [1], [2] while preserving valid numbers, dates, and brackets.

    Examples:
        - "Founded in 1529 [1]." -> "Founded in 1529."
        - "Rulers [1], [2] conquered..." -> "Rulers conquered..."
        - "Equation x[0] + 5" -> "Equation x[0] + 5" (preserved)
        - "[citation needed]" -> "[citation needed]" (preserved)
    """
    if not text:
        return ""

    # Strip bracketed pure integer numbers e.g. [1], [2], [ 12 ]
    cleaned = re.sub(r"\s*\[\s*\d+\s*\]", "", text)

    # Normalize any double spaces left behind
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    # Ensure punctuation hugs preceding words cleanly e.g. "word ." -> "word."
    cleaned = re.sub(r"\s+([,\.!?;:])", r"\1", cleaned)

    return cleaned.strip()


def format_answer_with_primary_reference(
    answer_text: str,
    primary_article: Optional[PrimaryArticle],
) -> str:
    """Format final answer presentation with a single trailing primary article hyperlink.

    Format:
        Answer paragraph... [Article Title](canonical_article_url)
    """
    cleaned_answer = strip_inline_citations(answer_text)

    if not primary_article or not primary_article.title or not primary_article.url:
        return cleaned_answer

    # Ensure clean trailing hyperlink on final paragraph
    hyperlink = f"[{primary_article.title.strip()}]({primary_article.url.strip()})"

    # If the answer already contains this exact link at the end, avoid duplication
    if cleaned_answer.endswith(hyperlink):
        return cleaned_answer

    return f"{cleaned_answer} {hyperlink}".strip()
