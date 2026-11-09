"""Single Primary Article Citation Manager for Nexia Answers.

Evaluates evidence passages and verified factual claims to select a single,
authoritative primary Wikipedia article reference, and formats the user-facing
answer with a clean trailing markdown hyperlink:

[Gupta Empire](https://en.wikipedia.org/wiki/Gupta_Empire)

with no inline numbered citations, raw URLs, middle-of-paragraph citations,
or separate bibliographies/Sources sections.
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

        - If a single article contributed: selected.
        - If multiple articles contributed: selects the strongest overall supporting article.
        - If no valid Wikipedia article is available: returns None (safe fallback).
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

            # Ignore ungrounded/unknown sources
            if not doc_title or doc_title == "Unknown Source":
                continue

            if not source_url and doc_title:
                source_url = f"https://en.wikipedia.org/wiki/{doc_title.replace(' ', '_')}"

            # Validate URL protocol
            if not source_url.startswith(("http://", "https://")):
                continue

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
                            # Require at least 2 significant words and at least 20% of sentence vocabulary
                            if matches >= 2 and (matches / len(s_words)) >= 0.20 and matches > max_matches:
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
            entity_bonus = 0.20 if (t_words & q_words) else 0.0

            # Composite coverage score: claim support dominates (50%), relevance (30%), subqueries (20%) + bonus
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

        # Sort candidates deterministically: highest composite score, then supported claims, then retrieval score
        candidate_scores.sort(
            key=lambda a: (a.coverage_score, a.supported_claims, a.retrieval_score, a.title),
            reverse=True,
        )

        best = candidate_scores[0]

        # Validation: Ensure the article is genuinely related to the query/passages
        # If it has 0 supported claims and 0 retrieval score, fallback safely
        if best.supported_claims == 0 and best.retrieval_score < 0.10:
            logger.info("Primary article selection fallback: Candidate '%s' has 0 support and low retrieval score.", best.title)
            return None

        # If coverage score is below threshold on weak/unrelated topics, fallback
        if best.coverage_score < coverage_threshold and best.supported_claims == 0:
            logger.info("Primary article selection fallback: Score %.2f < threshold %.2f and 0 claims.", best.coverage_score, coverage_threshold)
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
    """Remove internal citation markers e.g. [k], [1], [2], [1, 2], [source], [passage 1]
    and model-generated Sources/References sections while preserving legitimate square brackets
    (math intervals, editorial quotes) and Markdown hyperlinks.
    """
    if not text:
        return ""

    cleaned = text

    # 1. Strip model-generated trailing Sources / References / External links sections
    cleaned = re.sub(
        r"\n+(?:#+\s*)?(?:Sources|References|Citations|External [Ll]inks)\s*:?[\s\S]*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # Citation keyword pattern:
    kw = (
        r"(?:k|source|sources|ref|refs|reference|references|citation|citations|"
        r"passage|passages|doc|docs|document|documents|evidence|p\d+)"
        r"(?:\s*[:#]?\s*(?:\d+|[a-zA-Z0-9_-]+))*"
    )
    marker = rf"\[\s*(?:{kw}|\d+)\s*\](?!\()"

    # Chained or separated citation markers: e.g. [1], [2] or [k][1] or [1] and [2] or [k], [source]
    chained = rf"\s*{marker}(?:\s*(?:,|and|&)\s*{marker})*"
    cleaned = re.sub(chained, "", cleaned, flags=re.IGNORECASE)

    # Remaining standalone keywords e.g. [source 1]
    cleaned = re.sub(rf"\s*\[\s*{kw}\s*\](?!\()", "", cleaned, flags=re.IGNORECASE)

    # Remaining single integer citations e.g. [1]
    cleaned = re.sub(r"\s*\[\s*\d+\s*\](?!\()", "", cleaned)

    # Strip number ranges/lists [1, 2] or [1-3] unless preceded by math keywords
    def replace_multi_num(match):
        start = match.start()
        prefix = cleaned[max(0, start - 30):start].lower()
        if any(w in prefix for w in ["range", "interval", "between", "domain", "values in", "subset", "bounds", "matrix"]):
            return match.group(0)
        return ""

    cleaned = re.sub(r"\s*\[\s*\d+\s*(?:[,–-]\s*\d+\s*)+\s*\](?!\()", replace_multi_num, cleaned)

    # Clean punctuation anomalies:
    # 1. Trailing comma before punctuation: ',.' -> '.' or ',?' -> '?'
    cleaned = re.sub(r",\s*([\.!?;:])", r"\1", cleaned)
    # 2. Space before punctuation: ' .' -> '.'
    cleaned = re.sub(r"\s+([,\.!?;:])", r"\1", cleaned)
    # 3. Duplicate punctuation: '..' -> '.'
    cleaned = re.sub(r"([\.!?;:])\s*\1+", r"\1", cleaned)
    # 4. Strip raw URLs appearing alone on their own lines at the end
    cleaned = re.sub(r"\n+https?://\S+\s*$", "", cleaned)
    # 5. Normalize whitespace
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


class StreamingCitationFilter:
    """Incrementally filters internal citation markers like [k], [1], [source] during SSE token streaming
    without leaking partial tokens to the client, while preserving legitimate bracketed text and punctuation.
    """

    KW = (
        r"(?:k|source|sources|ref|refs|reference|references|citation|citations|"
        r"passage|passages|doc|docs|document|documents|evidence|p\d+)"
        r"(?:\s*[:#]?\s*(?:\d+|[a-zA-Z0-9_-]+))*"
    )
    MARKER_PATTERN = re.compile(rf"^\[\s*(?:{KW}|\d+)\s*\]$", re.IGNORECASE)

    def __init__(self):
        self.buffer = ""

    def process_chunk(self, chunk: str) -> str:
        self.buffer += chunk
        output = ""

        while True:
            bracket_idx = self.buffer.find("[")
            if bracket_idx == -1:
                # No bracket in buffer. Hold trailing whitespace/comma if at end of buffer
                # in case the subsequent chunk completes a citation marker
                stripped = self.buffer.rstrip(" \t,")
                if stripped:
                    output += stripped
                    self.buffer = self.buffer[len(stripped):]
                break

            pre_text = self.buffer[:bracket_idx]
            clean_pre = pre_text.rstrip(" \t,")
            pending_space = pre_text[len(clean_pre):]

            if clean_pre:
                output += clean_pre
                self.buffer = pending_space + self.buffer[bracket_idx:]
                bracket_idx = len(pending_space)

            close_idx = self.buffer.find("]", bracket_idx)
            if close_idx == -1:
                # Buffer has open '[' without ']'. If it exceeds threshold or has newline, flush it.
                if len(self.buffer) > 40 or "\n" in self.buffer:
                    output += self.buffer
                    self.buffer = ""
                break

            bracket_span = self.buffer[bracket_idx:close_idx + 1]
            after_bracket = self.buffer[close_idx + 1:]

            # Don't strip if this bracket is the anchor text of a markdown link [Title](url)
            if after_bracket.startswith("("):
                output += pending_space + bracket_span
                self.buffer = after_bracket
                continue

            if self.MARKER_PATTERN.match(bracket_span):
                # Detected internal citation marker: suppress it.
                # If chained to another marker via comma/and e.g. ', [2]'
                after_clean = re.sub(
                    rf"^(?:[ \t]*(?:,|and|&)[ \t]*)*(?=\[\s*(?:{self.KW}|\d+)\s*\])",
                    "",
                    after_bracket,
                    flags=re.IGNORECASE,
                )
                if after_clean != after_bracket:
                    self.buffer = after_clean
                elif after_bracket and after_bracket[0] in ".,!?;:":
                    self.buffer = after_bracket
                else:
                    self.buffer = (" " if pending_space.strip() == "" and pending_space else "") + after_bracket
            else:
                # Legitimate brackets (e.g. [10, 20] or [sic])
                output += pending_space + bracket_span
                self.buffer = after_bracket

        return output

    def flush(self) -> str:
        res = self.buffer
        self.buffer = ""
        return strip_inline_citations(res)


def format_answer_with_primary_reference(
    answer_text: str,
    primary_article: Optional[PrimaryArticle],
) -> str:
    """Format final answer presentation with exactly one trailing primary article hyperlink.

    Required output format:
        [Answer content paragraphs]

        [Article Title](https://en.wikipedia.org/wiki/Article_Title)

    Ensures no inline citations, no repeated links, and no duplicate final citations.
    """
    cleaned_answer = strip_inline_citations(answer_text)

    if not primary_article or not primary_article.title or not primary_article.url:
        return cleaned_answer

    title = primary_article.title.strip()
    url = primary_article.url.strip()
    hyperlink = f"[{title}]({url})"

    # Strip any trailing markdown link or repeated final reference if already present
    cleaned_answer = re.sub(r"\n*\[[^\]]+\]\(https?://[^\)]+\)\s*$", "", cleaned_answer).strip()

    # If the answer already contains this exact link at the end, clean it so we append consistently
    if cleaned_answer.endswith(hyperlink):
        cleaned_answer = cleaned_answer[:-len(hyperlink)].rstrip()

    return f"{cleaned_answer}\n\n{hyperlink}".strip()
