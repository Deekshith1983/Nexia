"""TF-IDF and Cosine Similarity Relevance Scorer with Term-Level Diagnostics."""

import math
from collections import Counter
from typing import List, Dict, Tuple, Set
from app.schemas.retrieval import Passage
from app.retrieval.query_processor import QueryProcessor


class TfidfRelevanceScorer:
    """Computes TF-IDF vectors, Cosine Similarity, and term overlap diagnostics between query and candidate passages."""

    def __init__(self, passages: List[Passage]):
        self.passages = passages
        self.num_passages = len(passages)
        self.processor = QueryProcessor()
        
        # Tokenize all passages
        self.passage_token_lists: List[List[str]] = [
            self.processor.tokenize(p.text) for p in passages
        ]
        
        # Compute Document Frequencies (DF) across the passage corpus
        self.doc_frequencies: Dict[str, int] = Counter()
        for token_list in self.passage_token_lists:
            unique_tokens = set(token_list)
            for t in unique_tokens:
                self.doc_frequencies[t] += 1

    def compute_idf(self, term: str) -> float:
        """Compute smooth Inverse Document Frequency: ln((1 + N) / (1 + DF)) + 1.0."""
        df = self.doc_frequencies.get(term, 0)
        return math.log((1.0 + self.num_passages) / (1.0 + df)) + 1.0

    def score_query(
        self, query: str
    ) -> List[Tuple[Passage, float, List[str], List[str]]]:
        """Score each passage against the query using TF-IDF cosine similarity.
        
        Returns a list of (Passage, score, matched_terms, unmatched_terms) sorted descending by score.
        """
        query_tokens = self.processor.tokenize(query)
        if not query_tokens or self.num_passages == 0:
            return [(p, 0.0, [], query_tokens) for p in self.passages]

        query_unique_terms: Set[str] = set(query_tokens)

        # 1. Compute Query TF-IDF Vector
        query_counts = Counter(query_tokens)
        query_total = len(query_tokens)
        query_vector: Dict[str, float] = {}
        query_norm_sq = 0.0

        for term, count in query_counts.items():
            tf = count / query_total
            idf = self.compute_idf(term)
            weight = tf * idf
            query_vector[term] = weight
            query_norm_sq += weight * weight

        query_norm = math.sqrt(query_norm_sq)
        if query_norm == 0.0:
            return [(p, 0.0, [], list(query_unique_terms)) for p in self.passages]

        # 2. Compute Passage TF-IDF Vectors & Cosine Similarities
        scored_passages: List[Tuple[Passage, float, List[str], List[str]]] = []

        for p, token_list in zip(self.passages, self.passage_token_lists):
            if not token_list:
                scored_passages.append((p, 0.0, [], list(query_unique_terms)))
                continue

            passage_token_set = set(token_list)
            matched_terms = sorted(list(query_unique_terms.intersection(passage_token_set)))
            unmatched_terms = sorted(list(query_unique_terms.difference(passage_token_set)))

            passage_counts = Counter(token_list)
            passage_total = len(token_list)
            dot_product = 0.0
            passage_norm_sq = 0.0

            for term, count in passage_counts.items():
                tf = count / passage_total
                idf = self.compute_idf(term)
                weight = tf * idf
                passage_norm_sq += weight * weight

                if term in query_vector:
                    dot_product += query_vector[term] * weight

            passage_norm = math.sqrt(passage_norm_sq)
            if passage_norm == 0.0:
                score = 0.0
            else:
                score = dot_product / (query_norm * passage_norm)

            # Round to 4 decimal places
            scored_passages.append((p, round(score, 4), matched_terms, unmatched_terms))

        # 3. Sort by score descending
        scored_passages.sort(key=lambda item: item[1], reverse=True)
        return scored_passages
