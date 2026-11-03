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

from app.schemas.retrieval import EntityItem, Stage4StructuredQuery, RelationshipIntent

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

    GENERIC_DESCRIPTORS: Set[str] = {
        "device", "ancient device", "machine", "ancient machine", "instrument", "tool",
        "system", "invention", "artifact", "mechanism", "person", "thing", "object",
        "structure", "building", "event", "phenomenon", "process", "concept", "element",
        "item", "material", "animal", "plant", "substance", "creature", "organism",
        "weapon", "ancient weapon", "monument", "astronomical events", "complex system",
        "system of gears", "complex system of gears", "gears"
    }

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
        # Normalize unicode dashes (en-dash, em-dash, figure dash, hyphen, minus sign)
        cleaned_q = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212–—]", " ", cleaned_q)
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
            # Strip leading determiners & interrogatives
            c_text = re.sub(r"^(?:the|a|an|their|his|her|its|what|which|whose|who)\s+", "", chunk.text.strip(), flags=re.IGNORECASE).strip()
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
            elif c_low in cls.GENERIC_DESCRIPTORS:
                sub_aspects.append(chunk)
            else:
                if not target_realm and len(chunk.split()) <= 4:
                    target_realm = chunk

        # 5. Determine Primary Entity
        # Prefer named entities (PERSON / GPE / ORG) or specific target_realm
        primary_entity = None
        if named_ents:
            primary_entity = named_ents[0][0]
        elif target_realm and target_realm.lower() not in cls.GENERIC_DESCRIPTORS:
            primary_entity = target_realm
        elif noun_chunks and noun_chunks[0].lower() not in cls.GENERIC_DESCRIPTORS:
            primary_entity = noun_chunks[0]

        # 6. Build Candidate EntityItems
        candidate_entities: List[EntityItem] = []
        seen_names = set()

        if primary_entity and primary_entity.lower() not in seen_names and primary_entity.lower() not in cls.GENERIC_DESCRIPTORS:
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

    @classmethod
    def analyze_stage4_intent(cls, question: str) -> Optional[Stage4StructuredQuery]:
        """Syntactic and dependency-based Stage 4 intent extraction using spaCy.

        Resolves relationship archetypes, directional roles, and named entities
        without relying on rigid regex string templates.
        """
        from app.retrieval.entity_extractor import EntityExtractor

        cleaned_q = question.strip()
        cleaned_q = re.sub(r"\bthr\b", "the", cleaned_q, flags=re.IGNORECASE)
        cleaned_q = re.sub(r"\bteh\b", "the", cleaned_q, flags=re.IGNORECASE)
        cleaned_q = re.sub(r"\bthier\b", "their", cleaned_q, flags=re.IGNORECASE)
        q_norm = re.sub(r"[\?\.\!]+$", "", cleaned_q).strip()

        nlp = cls.get_nlp()
        doc = nlp(q_norm)
        lemmas = [t.lemma_.lower() for t in doc]
        tokens_lower = [t.text.lower() for t in doc]
        text_lower = q_norm.lower()

        rel_type: Optional[str] = None
        direction = "OUTGOING"
        expected_type = "PERSON"
        q_type = "RELATIONSHIP"
        rel_terms: List[str] = []

        # 1. Classify Relationship Intent via Lemmatized Syntactic Actions
        # DIFFERENCE / COMPARISON
        if any(l in ["differ", "compare", "contrast"] for l in lemmas) or any(
            t in ["difference", "comparison", "contrast", "versus", "vs"] for t in tokens_lower
        ):
            rel_type = "DIFFERENCE_COMPARISON"
            direction = "BIDIRECTIONAL"
            expected_type = "COMPARISON"
            q_type = "MULTI_ENTITY_COMPARISON"
            rel_terms = ["difference", "comparison", "contrast", "versus", "vs"]

        # BILATERAL RELATIONS
        elif any(l in ["relate", "connect", "link"] for l in lemmas) or any(
            t in ["relationship", "relations", "connection", "history", "bilateral"] for t in tokens_lower
        ):
            rel_type = "BILATERAL_RELATIONS"
            direction = "BIDIRECTIONAL"
            expected_type = "RELATIONSHIP"
            q_type = "MULTI_ENTITY_RELATION"
            rel_terms = ["relationship", "relations", "connection", "history", "bilateral"]

        # FATHER OF THE NATION
        elif "father of the nation" in text_lower or "father of nation" in text_lower or "national founder" in text_lower:
            rel_type = "FATHER_OF_NATION_OF"
            direction = "INCOMING"
            expected_type = "PERSON"
            rel_terms = ["father of the nation", "father of nation", "rashtrapita", "national founder", "mahatma gandhi"]

        # ORDINAL OFFICE HOLDER (3rd prime minister, first president, etc.)
        elif any(
            t.ent_type_ == "ORDINAL" or t.text.lower() in [
                "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"
            ]
            for t in doc
        ) and any(
            k in text_lower for k in ["prime minister", "president", "king", "emperor", "ruler", "monarch", "chancellor"]
        ):
            rel_type = "ORDINAL_OFFICE_HOLDER"
            direction = "INCOMING"
            expected_type = "PERSON"
            rel_terms = ["prime minister", "office", "term", "order", "tenure"]

        # OFFICE HOLDERS / CATEGORY LISTING
        elif any(l in ["list", "name", "enumerate", "give", "show", "tell"] for l in lemmas) or (
            any(t in ["who", "what"] for t in tokens_lower)
            and any(
                k in text_lower for k in [
                    "states", "countries", "districts", "rulers", "presidents",
                    "monarchs", "kings", "emperors", "prime ministers", "provinces"
                ]
            )
        ):
            rel_type = "OFFICE_HOLDERS_OF"
            direction = "INCOMING"
            expected_type = "ENTITY_LIST"
            rel_terms = ["list", "table", "states", "territories", "countries", "office", "term", "birth place", "rule period"]

        # SUCCESSION (succeeded, successor, ruler after...)
        elif any(l in ["succeed"] for l in lemmas) or any(
            t in ["successor", "succession"] for t in tokens_lower
        ) or (
            "after" in tokens_lower and any(
                r in tokens_lower for r in ["ruler", "king", "emperor", "person", "leader", "who"]
            )
        ):
            rel_type = "SUCCESSOR_OF"
            direction = "OUTGOING"
            expected_type = "PERSON"
            rel_terms = ["successor", "succeeded", "succeeding", "succession", "heir", "after"]

        # PREDECESSOR (preceded, predecessor, ruler before...)
        elif any(l in ["precede"] for l in lemmas) or any(
            t in ["predecessor", "previous"] for t in tokens_lower
        ) or (
            "before" in tokens_lower and any(
                r in tokens_lower for r in ["ruler", "king", "emperor", "person", "leader", "who"]
            )
        ):
            rel_type = "PREDECESSOR_OF"
            direction = "INCOMING"
            expected_type = "PERSON"
            rel_terms = ["predecessor", "preceded", "preceding", "before", "previous"]

        # FOUNDED BY (founded, founder, established by, created by...)
        elif any(l in ["found", "establish", "create", "start"] for l in lemmas) or any(
            t in ["founder", "co-founder", "creator", "establisher"] for t in tokens_lower
        ):
            rel_type = "FOUNDED_BY"
            direction = "INCOMING"
            expected_type = "PERSON"
            rel_terms = ["founded", "founder", "co-founder", "established", "creator", "started"]

        # BORN IN (born, birthplace...)
        elif any(l in ["bear", "born"] for l in lemmas) or "birthplace" in text_lower or "born in" in text_lower:
            rel_type = "BORN_IN"
            direction = "OUTGOING"
            expected_type = "LOCATION"
            rel_terms = ["born", "birthplace", "native", "place of birth"]

        # WORKED WITH (worked with, partnered with, collaborated with...)
        elif any(l in ["work", "partner", "collaborate", "associate"] for l in lemmas):
            rel_type = "WORKED_WITH"
            direction = "BIDIRECTIONAL"
            expected_type = "PERSON"
            rel_terms = ["worked with", "partner", "colleague", "collaborator", "associated"]

        if not rel_type:
            return None

        # 2. Extract Named Entities using spaCy NER + Noun Chunks
        OFFICE_FILTER = {
            "prime minister", "president", "king", "emperor", "ruler", "monarch",
            "successor", "predecessor", "founder", "states", "countries", "districts",
            "rule period", "difference", "relationship", "father", "nation", "office"
        }
        extracted_names: List[str] = []
        seen = set()

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG", "LOC", "NORP"] and ent.text.lower() not in cls.STOPWORDS_CHUNKS:
                c_text = ent.text.strip(".,!? ")
                low = c_text.lower()
                if c_text and low not in seen and low not in OFFICE_FILTER:
                    seen.add(low)
                    extracted_names.append(c_text)

        for chunk in doc.noun_chunks:
            c_text = chunk.text.strip(".,!? ")
            c_clean = re.sub(r"^(?:the|a|an|their|his|her|its)\s+", "", c_text, flags=re.IGNORECASE).strip()
            low = c_clean.lower()
            if low not in seen and low not in cls.STOPWORDS_CHUNKS and low not in OFFICE_FILTER and len(c_clean) > 2:
                if low in EntityExtractor.KNOWN_COUNTRIES or any(term in low for term in ["empire", "dynasty", "kingdom", "company", "inc", "corp", "university", "institute"]):
                    seen.add(low)
                    extracted_names.append(c_clean)

        if not extracted_names:
            return None

        # 3. Assemble Entities with Roles
        entities: List[EntityItem] = []
        if rel_type in ["BILATERAL_RELATIONS", "DIFFERENCE_COMPARISON"] and len(extracted_names) >= 2:
            ent1 = extracted_names[0]
            ent2 = extracted_names[1]
            norm1 = EntityExtractor.normalize_entity_text(ent1)
            norm2 = EntityExtractor.normalize_entity_text(ent2)
            entities = [
                EntityItem(text=ent1, normalized_text=norm1, type=EntityExtractor._infer_entity_type(norm1), role="SUBJECT_A"),
                EntityItem(text=ent2, normalized_text=norm2, type=EntityExtractor._infer_entity_type(norm2), role="SUBJECT_B"),
            ]
            retrieval_terms = [norm1, norm2] + rel_terms
        else:
            primary = extracted_names[0]
            norm_primary = EntityExtractor.normalize_entity_text(primary)
            entities = [
                EntityItem(text=primary, normalized_text=norm_primary, type=EntityExtractor._infer_entity_type(norm_primary), role="SUBJECT")
            ]
            if rel_type == "OFFICE_HOLDERS_OF":
                # Find category noun (states, presidents, rulers, etc.)
                cat = "rulers"
                for kw in ["states", "countries", "districts", "rulers", "presidents", "monarchs", "kings", "emperors", "prime ministers"]:
                    if kw in text_lower:
                        cat = kw
                        break
                retrieval_terms = [
                    f"List of {cat} of {norm_primary}",
                    f"List of {cat} in {norm_primary}",
                    f"{cat} of {norm_primary}",
                    f"{cat} in {norm_primary}",
                    norm_primary,
                    cat,
                ] + rel_terms
            elif rel_type == "ORDINAL_OFFICE_HOLDER":
                retrieval_terms = [norm_primary, f"List of {norm_primary}"] + rel_terms
            else:
                retrieval_terms = [norm_primary] + rel_terms

        return Stage4StructuredQuery(
            raw_question=question,
            question_type=q_type,
            entities=entities,
            relation=RelationshipIntent(
                type=rel_type,
                direction=direction,
                target_role="TARGET_OBJECT" if q_type == "RELATIONSHIP" else "SUBJECT_B",
                relation_terms=rel_terms,
            ),
            expected_answer_type=expected_type,
            retrieval_terms=retrieval_terms,
        )

    @classmethod
    def is_list_question(cls, question: str) -> bool:
        """Detect if the user query is asking for an enumeration, list, or inventory."""
        q_low = question.lower().strip()
        LIST_PREFIXES = ("list", "name", "give", "enumerate", "what are", "who are", "show me", "tell me", "mention")
        if any(q_low.startswith(p) for p in LIST_PREFIXES):
            return True
        try:
            doc = cls.get_nlp()(question.strip())
            for token in doc[:3]:
                if token.pos_ == "VERB" and token.lemma_ in ["list", "name", "enumerate", "give", "mention"]:
                    return True
        except Exception:
            pass
        return False
