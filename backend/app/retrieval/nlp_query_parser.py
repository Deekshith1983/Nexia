"""Semantic NLP Query Parser using lightweight local spaCy engine.

Extracts grammatical noun chunks, named entities (PERSON, GPE, ORG, NORP, EVENT),
syntactic dependencies, and detects follow-up pronoun references without external cloud calls.
"""

import logging
import re
from typing import List, Optional, Set, Tuple
from pydantic import BaseModel, Field

import spacy
from spacy.language import Language

from app.schemas.retrieval import EntityItem

logger = logging.getLogger(__name__)


class ParsedSemanticIntent(BaseModel):
    """Structured semantic interpretation of a user query."""

    raw_question: str
    cleaned_question: str
    named_entities: List[Tuple[str, str]] = Field(default_factory=list)  # (text, label)
    candidate_entities: List[EntityItem] = Field(default_factory=list)
    primary_entity: Optional[str] = None
    role_or_office: Optional[str] = None
    sub_aspects: List[str] = Field(default_factory=list)
    retrieval_terms: List[str] = Field(default_factory=list)
    is_follow_up: bool = False
    anaphoric_pronouns: List[str] = Field(default_factory=list)


class NLPQueryParser:
    """Lightweight local CPU NLP query analyzer using spaCy en_core_web_sm."""

    _nlp_instance: Optional[Language] = None

    STOPWORDS_CHUNKS: Set[str] = {
        "who", "what", "where", "when", "why", "how", "which", "someone", "anyone",
        "something", "anything", "list", "name", "names", "details", "information",
        "tell", "give", "show", "describe", "explain", "find", "enumerate", "provide"
    }

    PRONOUNS: Set[str] = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs", "it", "its"}

    @classmethod
    def get_nlp(cls) -> Language:
        """Singleton loader for spaCy small English pipeline."""
        if cls._nlp_instance is None:
            try:
                cls._nlp_instance = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning("Could not load spacy en_core_web_sm directly: %s. Loading blank English pipeline.", e)
                cls._nlp_instance = spacy.blank("en")
        return cls._nlp_instance

    @classmethod
    def parse(cls, question: str) -> ParsedSemanticIntent:
        """Analyze natural language query into structured semantic intent."""
        cleaned_q = question.strip()
        # Clean common user typing quirks
        cleaned_q = re.sub(r"\bthr\b", "the", cleaned_q, flags=re.IGNORECASE)
        cleaned_q = re.sub(r"\bteh\b", "the", cleaned_q, flags=re.IGNORECASE)
        cleaned_q = re.sub(r"\bthier\b", "their", cleaned_q, flags=re.IGNORECASE)
        q_norm = re.sub(r"[\?\.\!]+$", "", cleaned_q).strip()

        nlp = cls.get_nlp()
        doc = nlp(q_norm)

        # 1. Extract Named Entities from NER
        named_ents: List[Tuple[str, str]] = []
        for ent in doc.ents:
            clean_ent = ent.text.strip()
            if clean_ent.lower() not in cls.STOPWORDS_CHUNKS and len(clean_ent) > 1:
                named_ents.append((clean_ent, ent.label_))

        # 2. Extract Meaningful Noun Chunks
        noun_chunks: List[str] = []
        for chunk in doc.noun_chunks:
            # Strip leading determiners
            c_text = re.sub(r"^(?:the|a|an|their|his|her|its)\s+", "", chunk.text.strip(), flags=re.IGNORECASE).strip()
            if c_text.lower() not in cls.STOPWORDS_CHUNKS and len(c_text) > 1:
                noun_chunks.append(c_text)

        # 3. Detect Anaphoric Pronouns (Follow-up detection)
        detected_pronouns: List[str] = []
        for token in doc:
            if token.text.lower() in cls.PRONOUNS and token.pos_ == "PRON":
                detected_pronouns.append(token.text.lower())

        # 4. Extract Syntactic Roles (e.g. "rulers", "prime ministers", "founder")
        role_or_office = None
        target_realm = None
        sub_aspects = []

        ROLE_KEYWORDS = {
            "ruler", "rulers", "king", "kings", "emperor", "emperors", "monarch", "monarchs",
            "prime minister", "prime ministers", "president", "presidents", "chancellor",
            "founder", "founders", "creator", "successor", "predecessor", "father of nation",
            "father of the nation", "national founder"
        }

        ASPECT_KEYWORDS = {
            "period", "rule period", "time", "period of time", "dates", "years", "reign",
            "contribution", "contributions", "education", "birth", "birth date", "birth place",
            "achievements", "battles", "history", "administration", "policy"
        }

        for chunk in noun_chunks:
            c_low = chunk.lower()
            if any(rk in c_low for rk in ROLE_KEYWORDS):
                if not role_or_office:
                    role_or_office = chunk
            elif any(ak in c_low for ak in ASPECT_KEYWORDS):
                sub_aspects.append(chunk)
            else:
                if not target_realm and len(chunk.split()) <= 4:
                    target_realm = chunk

        # 5. Determine Primary Entity
        # Prefer target_realm or named entities (PERSON / GPE / ORG)
        primary_entity = None
        if target_realm:
            primary_entity = target_realm
        elif named_ents:
            primary_entity = named_ents[0][0]
        elif noun_chunks:
            primary_entity = noun_chunks[0]

        # 6. Build Candidate EntityItems
        candidate_entities: List[EntityItem] = []
        seen_names = set()

        if primary_entity and primary_entity.lower() not in seen_names:
            seen_names.add(primary_entity.lower())
            ent_type = "GPE_COUNTRY" if any(l in ["GPE", "LOC"] for t, l in named_ents if t == primary_entity) else "PERSON"
            candidate_entities.append(
                EntityItem(
                    text=primary_entity,
                    normalized_text=primary_entity.title(),
                    type=ent_type,
                    role="SUBJECT",
                )
            )

        for ent_text, ent_label in named_ents:
            if ent_text.lower() not in seen_names:
                seen_names.add(ent_text.lower())
                candidate_entities.append(
                    EntityItem(
                        text=ent_text,
                        normalized_text=ent_text.title(),
                        type="GPE_COUNTRY" if ent_label in ["GPE", "LOC"] else "PERSON",
                        role="RELATED_ENTITY",
                    )
                )

        # 7. Formulate Retrieval Terms
        retrieval_terms: List[str] = []
        if primary_entity:
            retrieval_terms.append(primary_entity.title())
        if role_or_office and primary_entity:
            retrieval_terms.append(f"{role_or_office} of {primary_entity.title()}")
            retrieval_terms.append(f"List of {role_or_office} of {primary_entity.title()}")
        if role_or_office:
            retrieval_terms.append(role_or_office)
        for asp in sub_aspects:
            retrieval_terms.append(asp)

        is_follow_up = bool(detected_pronouns and not primary_entity)

        return ParsedSemanticIntent(
            raw_question=question,
            cleaned_question=q_norm,
            named_entities=named_ents,
            candidate_entities=candidate_entities,
            primary_entity=primary_entity.title() if primary_entity else None,
            role_or_office=role_or_office,
            sub_aspects=sub_aspects,
            retrieval_terms=retrieval_terms,
            is_follow_up=is_follow_up,
            anaphoric_pronouns=detected_pronouns,
        )
