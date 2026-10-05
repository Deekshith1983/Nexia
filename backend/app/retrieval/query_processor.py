"""Query Processor: Normalizes, tokenizes, and filters questions and passage text."""

import re
from typing import List, Set


class QueryProcessor:
    """Handles text normalization, tokenization, and stopword filtering for lexical retrieval."""

    # Standard English stopwords commonly appearing in questions and text
    STOPWORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
        "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
        "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
        "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
        "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
        "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
        "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
        "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
        "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
        "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
        "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
        "they're", "they've", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
        "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
        "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
        "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
        "your", "yours", "yourself", "yourselves",
    }

    # Common question prefix patterns to help identify subject entities
    QUESTION_PREFIX_REGEX = re.compile(
        r"^(where\s+(?:did|was|is|are|were|can)|what\s+(?:is|was|are|were|did|does|do)|"
        r"when\s+(?:did|was|is|are|were)|who\s+(?:was|is|are|were)|how\s+(?:did|was|is|are|were|does)|"
        r"why\s+(?:did|was|is|are|were|does))\s+",
        re.IGNORECASE,
    )

    @classmethod
    def tokenize(cls, text: str, remove_stopwords: bool = True) -> List[str]:
        """Normalize text and split into alphanumeric word tokens."""
        if not text:
            return []

        # Convert to lowercase and find all alphanumeric tokens (including numbers and hyphens)
        raw_tokens = re.findall(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b", text.lower())

        if not remove_stopwords:
            return raw_tokens

        filtered = [t for t in raw_tokens if t not in cls.STOPWORDS and len(t) > 1]
        # If removing stopwords would result in empty tokens, retain raw tokens
        return filtered if filtered else raw_tokens

    @classmethod
    def extract_search_terms_from_question(cls, question: str) -> str:
        """Strip question prefixes to produce candidate search terms if no explicit document title is given."""
        cleaned = question.strip()
        stripped = cls.QUESTION_PREFIX_REGEX.sub("", cleaned)
        # Remove trailing question mark
        stripped = re.sub(r"\?+$", "", stripped).strip()
        return stripped if stripped else cleaned
