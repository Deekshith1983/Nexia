"""BM25 (Best Matching 25) Passage Relevance Scorer."""

import math
from collections import Counter
from typing import List, Dict, Tuple, Set
from app.schemas.retrieval import Passage
from app.retrieval.query_processor import QueryProcessor


class BM25RelevanceScorer:
    """Computes BM25 relevance scores with term frequency saturation and document length normalization."""

    def __init__(self, passages: List[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.num_passages = len(passages)
        self.k1 = k1
        self.b = b
        self.processor = QueryProcessor()

        # Tokenize passages and compute lengths
        self.passage_token_lists: List[List[str]] = [
            self.processor.tokenize(p.text) for p in passages
        ]
        self.doc_lengths: List[int] = [len(tokens) for tokens in self.passage_token_lists]
        self.avg_doc_length: float = (
            sum(self.doc_lengths) / self.num_passages if self.num_passages > 0 else 1.0
        )

        # Compute Document Frequencies (DF) across the passage corpus
        self.doc_frequencies: Dict[str, int] = Counter()
        for token_list in self.passage_token_lists:
            for t in set(token_list):
                self.doc_frequencies[t] += 1

    def compute_idf(self, term: str) -> float:
        """Compute Robertson-Spärck Jones BM25 IDF: ln((N - DF + 0.5) / (DF + 0.5) + 1)."""
        df = self.doc_frequencies.get(term, 0)
        return math.log((self.num_passages - df + 0.5) / (df + 0.5) + 1.0)

    def score_query(
        self, query_tokens: List[str]
    ) -> List[Tuple[Passage, float, List[str], List[str]]]:
        """Score all passages against query tokens using BM25.
        
        Returns a list of (Passage, score, matched_terms, unmatched_terms) sorted descending by score.
        """
        if not query_tokens or self.num_passages == 0:
            return [(p, 0.0, [], query_tokens) for p in self.passages]

        query_unique: Set[str] = set(query_tokens)
        raw_scores: List[float] = []
        matched_unmatched_list: List[Tuple[List[str], List[str]]] = []

        for p_idx, token_list in enumerate(self.passage_token_lists):
            if not token_list:
                raw_scores.append(0.0)
                matched_unmatched_list.append(([], list(query_unique)))
                continue

            doc_len = self.doc_lengths[p_idx]
            counts = Counter(token_list)
            token_set = set(token_list)

            matched = sorted(list(query_unique.intersection(token_set)))
            unmatched = sorted(list(query_unique.difference(token_set)))
            matched_unmatched_list.append((matched, unmatched))

            score = 0.0
            for term in query_unique:
                if term in counts:
                    freq = counts[term]
                    idf = self.compute_idf(term)
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
                    score += idf * (numerator / denominator)

            raw_scores.append(score)

        # Normalize BM25 scores relative to max for easy threshold comparison
        max_score = max(raw_scores) if raw_scores else 1.0
        normalized_scores = [
            round(s / max_score, 4) if max_score > 0 else 0.0 for s in raw_scores
        ]

        scored_passages: List[Tuple[Passage, float, List[str], List[str]]] = []
        for p, norm_score, (matched, unmatched) in zip(
            self.passages, normalized_scores, matched_unmatched_list
        ):
            scored_passages.append((p, norm_score, matched, unmatched))

        scored_passages.sort(key=lambda item: item[1], reverse=True)
        return scored_passages
