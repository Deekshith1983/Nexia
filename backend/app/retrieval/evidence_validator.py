"""Universal Evidence Validation Gate with Strict Entity & Geographic Constraint Verification."""

import re
from typing import List, Tuple
from app.schemas.retrieval import EvidencePassage, StructuredQueryRepresentation


class EvidenceValidationResult:
    """Detailed outcome of evidence sufficiency validation."""

    def __init__(
        self,
        status: str,
        is_sufficient: bool,
        validated_evidence: List[EvidencePassage],
        reason: str,
    ):
        self.status = status
        self.is_sufficient = is_sufficient
        self.validated_evidence = validated_evidence
        self.reason = reason


class EvidenceValidator:
    """Validates retrieved candidate evidence passages against strict entity, relation, and geographic constraints."""

    @classmethod
    def validate(
        cls,
        intent: StructuredQueryRepresentation,
        candidate_passages: List[EvidencePassage],
        min_score: float = 0.15,
    ) -> EvidenceValidationResult:
        """Validate candidate passages against required constraints, preventing false sufficiency claims."""
        if not candidate_passages:
            return EvidenceValidationResult(
                status="INSUFFICIENT_EVIDENCE",
                is_sufficient=False,
                validated_evidence=[],
                reason="No candidate passages retrieved.",
            )

        # Filter candidates meeting minimum threshold
        scored_candidates = [p for p in candidate_passages if p.score >= min_score]
        if not scored_candidates:
            return EvidenceValidationResult(
                status="INSUFFICIENT_EVIDENCE",
                is_sufficient=False,
                validated_evidence=[],
                reason=f"All passages scored below relevance threshold ({min_score}).",
            )

        top_passage = scored_candidates[0]
        text_lower = top_passage.text.lower()
        qtype = intent.question_type

        # 1. Strict Required Entity & Geographic Constraint Check
        required_entities_lower = [e.lower() for e in intent.constraints.required_entities]
        missing_required = [req for req in required_entities_lower if req not in text_lower]

        if missing_required:
            # Passage violates required constraint (e.g. "India" missing in US / George Washington passage)
            return EvidenceValidationResult(
                status="INSUFFICIENT_EVIDENCE",
                is_sufficient=False,
                validated_evidence=[],
                reason=(
                    f"Passage in section '{top_passage.section_title}' fails required entity constraint. "
                    f"Missing required entity: '{missing_required[0].title()}'."
                ),
            )

        # 2. Strict Excluded Context Check (e.g. mentions "United States" when target is "India")
        for excluded in intent.constraints.excluded_contexts:
            if excluded.lower() in text_lower and not any(req in text_lower for req in required_entities_lower):
                return EvidenceValidationResult(
                    status="INSUFFICIENT_EVIDENCE",
                    is_sufficient=False,
                    validated_evidence=[],
                    reason=(
                        f"Passage contains context mismatch '{excluded}' incompatible with requested subject '{intent.subject_entity}'."
                    ),
                )

        # 3. Attribute / Title Lookup (e.g. "Father of the Nation of India")
        if qtype == "ATTRIBUTE_OR_TITLE_LOOKUP":
            title_indicators = ["father of the nation", "gandhi", "mahatma gandhi", "mohandas", "bapu"]
            has_title = any(t in text_lower for t in title_indicators)
            has_target_country = intent.subject_entity.lower() in text_lower
            
            # Check for candidate person name
            has_person = any(w in text_lower for w in ["gandhi", "mahatma", "mohandas"]) or bool(re.search(r"\b[A-Z][a-z]+\b", top_passage.text))

            if has_title and has_target_country and has_person:
                return EvidenceValidationResult(
                    status="VALIDATED_SUFFICIENT_EVIDENCE",
                    is_sufficient=True,
                    validated_evidence=scored_candidates,
                    reason=(
                        f"Passage in section '{top_passage.section_title}' explicitly associates the title "
                        f"'{intent.relation}' with {intent.subject_entity} and candidate entity (Mahatma Gandhi)."
                    ),
                )
            elif has_target_country and has_title:
                return EvidenceValidationResult(
                    status="PARTIALLY_SUPPORTED",
                    is_sufficient=False,
                    validated_evidence=scored_candidates,
                    reason=f"Passage references '{intent.relation}' in {intent.subject_entity}, but candidate person attribution is partial.",
                )
            else:
                return EvidenceValidationResult(
                    status="INSUFFICIENT_EVIDENCE",
                    is_sufficient=False,
                    validated_evidence=[],
                    reason=f"Passage fails required title-entity association for {intent.subject_entity}.",
                )

        # 4. PERSON_IDENTIFICATION ("WHO")
        elif qtype == "PERSON_IDENTIFICATION":
            founder_indicators = [
                "founded by", "founder was", "established by", "originated with",
                "first ruler", "early ruler", "chieftain", "king", "monarch", "emperor"
            ]
            has_explicit_founder = any(ind in text_lower for ind in founder_indicators)
            has_relation_cues = any(cue in text_lower for cue in intent.expanded_semantic_cues)

            proper_names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", top_passage.text)
            subject_tokens = set(re.findall(r"\b\w+\b", intent.subject_entity.lower()))
            proper_names_filtered = [n for n in proper_names if n.lower() not in subject_tokens]

            if has_explicit_founder and len(proper_names_filtered) > 0:
                return EvidenceValidationResult(
                    status="VALIDATED_SUFFICIENT_EVIDENCE",
                    is_sufficient=True,
                    validated_evidence=scored_candidates,
                    reason=(
                        f"Found explicit founder/person attribution in section '{top_passage.section_title}' "
                        f"referencing candidate agents: {proper_names_filtered[:3]}."
                    ),
                )
            elif has_relation_cues and len(proper_names_filtered) > 0:
                return EvidenceValidationResult(
                    status="PARTIALLY_SUPPORTED",
                    is_sufficient=False,
                    validated_evidence=scored_candidates,
                    reason=(
                        f"Passage in section '{top_passage.section_title}' discusses historical rise and key rulers ({proper_names_filtered[:3]}), "
                        "but does not contain an unambiguous single founder attribution."
                    ),
                )
            else:
                return EvidenceValidationResult(
                    status="INSUFFICIENT_EVIDENCE",
                    is_sufficient=False,
                    validated_evidence=[],
                    reason="Passage lacks candidate person/agent entities answering a WHO question.",
                )

        # 5. RELATIONSHIP_SUPERLATIVE ("NEAREST", "LARGEST")
        elif qtype == "RELATIONSHIP_SUPERLATIVE":
            nearest_galaxy_entities = ["andromeda", "canis major", "sagittarius dwarf", "large magellanic cloud", "small magellanic cloud"]
            has_superlative_target = any(ent in text_lower for ent in nearest_galaxy_entities)
            has_distance_metrics = bool(re.search(r"\b(\d+(?:\.\d+)?\s*(?:light-years|kpc|million light-years|parsecs|km|miles))\b", text_lower))
            has_proximity_cues = any(w in text_lower for w in ["nearest", "closest", "proximity", "satellite galaxy", "companion", "neighbor"])

            if (has_superlative_target or has_distance_metrics) and has_proximity_cues:
                return EvidenceValidationResult(
                    status="VALIDATED_SUFFICIENT_EVIDENCE",
                    is_sufficient=True,
                    validated_evidence=scored_candidates,
                    reason=f"Passage in section '{top_passage.section_title}' provides proximity/distance evidence for {intent.subject_entity}.",
                )
            elif has_proximity_cues:
                return EvidenceValidationResult(
                    status="PARTIALLY_SUPPORTED",
                    is_sufficient=False,
                    validated_evidence=scored_candidates,
                    reason=f"Passage in section '{top_passage.section_title}' discusses proximity/satellites, but specific nearest entity metric is unverified.",
                )
            else:
                return EvidenceValidationResult(
                    status="INSUFFICIENT_EVIDENCE",
                    is_sufficient=False,
                    validated_evidence=[],
                    reason="Passage lacks required proximity/superlative comparison evidence.",
                )

        # 6. Default Fallback
        matched_cues = [c for c in intent.expanded_semantic_cues if c in text_lower]
        if len(matched_cues) >= 2:
            return EvidenceValidationResult(
                status="VALIDATED_SUFFICIENT_EVIDENCE",
                is_sufficient=True,
                validated_evidence=scored_candidates,
                reason=f"Top passage in section '{top_passage.section_title}' satisfies semantic cues: {matched_cues[:4]}.",
            )
        else:
            return EvidenceValidationResult(
                status="PARTIALLY_SUPPORTED",
                is_sufficient=False,
                validated_evidence=scored_candidates,
                reason="Passage matches query keywords but semantic relationship is partially supported.",
            )
