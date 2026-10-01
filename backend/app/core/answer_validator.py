import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from app.core.llm_interface import BaseLanguageModel
from app.services.wikipedia_retriever import RetrievalResult, RetrievalStatus
from app.knowledge.models import Claim, GroundingStatus
from app.optimization.cache import local_cache_manager
from app.optimization.telemetry import telemetry_tracker

logger = logging.getLogger("nexia.validator")


class ValidationStatus:
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class ValidationResult(BaseModel):
    """Structured representation of answer validation against evidence with claim-level provenance."""
    status: str  # ValidationStatus
    reason: Optional[str] = None
    grounding_status: Optional[str] = None
    question_satisfied: bool = True
    answer_shape_satisfied: bool = True
    claims: List[Claim] = Field(default_factory=list)
    unsupported_claims: List[Claim] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Stop-words filtered out before overlap scoring ───────────────────────────
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "to", "of", "in",
    "on", "at", "by", "for", "with", "from", "as", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "he", "she", "his",
    "her", "we", "us", "our", "you", "your", "i", "me", "my", "and",
    "but", "or", "not", "also", "so", "than", "more", "most", "very",
    "such", "about", "which", "who", "what", "when", "where", "how",
}


def _significant_words(text: str) -> Set[str]:
    """Extract significant (non-stop-word) lowercase alphabetic tokens."""
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def _check_contract_satisfaction(cleaned_answer: str, answer_contract: Optional[Any]) -> Tuple[bool, Optional[str]]:
    """Verify whether the candidate answer structurally and semantically addresses the AnswerContract."""
    if not answer_contract:
        return True, None

    exp_type = getattr(answer_contract, "expected_type", None) or getattr(answer_contract, "answer_type", None)
    exp_type_str = getattr(exp_type, "value", str(exp_type)).lower() if exp_type else ""
    req_rel = (getattr(answer_contract, "requested_relation", None) or "").lower()
    target_ent = (getattr(answer_contract, "target_entity", None) or "").lower()
    concept_a = (getattr(answer_contract, "concept_a", None) or "").lower()
    concept_b = (getattr(answer_contract, "concept_b", None) or "").lower()
    ans_lower = cleaned_answer.lower()

    # 1. LIST Contract: must have multiple items/lines or bullet points
    if "list" in exp_type_str or req_rel in ("rulers", "kings", "monarchs", "prime ministers", "dynasties"):
        lines = [line.strip() for line in cleaned_answer.split("\n") if line.strip()]
        has_bullet = any(re.match(r"^[-*•\d]+[\.\)]?\s+", line) for line in lines)
        has_colon_group = any(":" in line for line in lines)
        has_multiple_lines = len(lines) >= 2
        if not (has_bullet or (has_colon_group and has_multiple_lines) or len(lines) >= 3):
            return False, "LIST requirement not satisfied: answer must provide structured list items or chronological lines."

    # 2. Conceptual Distinction Contract: must cover Concept A and Concept B with distinction rationale
    elif req_rel == "conceptual_distinction" or (concept_a and concept_b):
        c_a = concept_a if concept_a else "empire"
        c_b = concept_b if concept_b else "dynasty"
        has_a = c_a in ans_lower
        has_b = c_b in ans_lower or "dynasties" in ans_lower
        if not (has_a and has_b):
            return False, f"Conceptual distinction requirement not satisfied: answer must address both '{c_a}' and '{c_b}'."

    # 3. CAUSAL Contract: must contain causal explanation connectives
    elif "causal" in exp_type_str or req_rel in ("causes", "cause", "decline", "fall", "collapse"):
        has_causal_terms = any(
            t in ans_lower
            for t in ["because", "due to", "resulted in", "led to", "caused by", "reason", "factors", "following", "after the fall", "defeat", "decline"]
        )
        if not has_causal_terms:
            return False, "CAUSAL requirement not satisfied: answer must explain causes/reasons rather than isolated dates."

    # 4. RELATION / ATTRIBUTE Contract: target entity reference
    if target_ent and len(target_ent) >= 3:
        target_words = set(re.findall(r"\b[a-z]{3,}\b", target_ent))
        if target_words and not any(w in ans_lower for w in target_words):
            return False, f"Relation requirement not satisfied: answer must reference target entity '{target_ent}'."

    return True, None


def _heuristic_validate(
    answer: str,
    evidence_content: str,
    answer_contract: Optional[Any] = None,
) -> ValidationResult:
    """
    Fast deterministic validation using significant-word overlap and contract checks.
    """
    answer_words = _significant_words(answer)
    evidence_words = _significant_words(evidence_content)

    if not answer_words:
        return ValidationResult(
            status=ValidationStatus.INSUFFICIENT_EVIDENCE,
            reason="Answer contains no significant content words.",
            question_satisfied=False,
            answer_shape_satisfied=False,
            metadata={"method": "heuristic"},
        )

    if not evidence_words:
        return ValidationResult(
            status=ValidationStatus.INSUFFICIENT_EVIDENCE,
            reason="Evidence contains no significant content words.",
            question_satisfied=False,
            answer_shape_satisfied=False,
            metadata={"method": "heuristic"},
        )

    overlap_words = answer_words & evidence_words
    overlap_ratio = len(overlap_words) / len(answer_words)

    logger.debug(
        "[VALIDATOR] answer_words=%d  evidence_words=%d  overlap=%d  ratio=%.2f",
        len(answer_words), len(evidence_words), len(overlap_words), overlap_ratio,
    )

    shape_ok, shape_reason = _check_contract_satisfaction(answer, answer_contract)

    if overlap_ratio >= 0.20 and shape_ok:
        return ValidationResult(
            status=ValidationStatus.SUPPORTED,
            reason=f"Word overlap {overlap_ratio:.0%} ({len(overlap_words)}/{len(answer_words)} significant terms).",
            question_satisfied=True,
            answer_shape_satisfied=True,
            metadata={
                "method": "heuristic",
                "overlap_ratio": round(overlap_ratio, 3),
                "overlap_count": len(overlap_words),
                "answer_word_count": len(answer_words),
            },
        )
    elif overlap_ratio >= 0.20 and not shape_ok:
        return ValidationResult(
            status=ValidationStatus.PARTIALLY_SUPPORTED,
            reason=shape_reason or "Answer text is grounded but does not satisfy the AnswerContract shape/relation.",
            question_satisfied=False,
            answer_shape_satisfied=False,
            metadata={
                "method": "heuristic",
                "overlap_ratio": round(overlap_ratio, 3),
                "overlap_count": len(overlap_words),
                "answer_word_count": len(answer_words),
            },
        )
    else:
        return ValidationResult(
            status=ValidationStatus.INSUFFICIENT_EVIDENCE,
            reason=f"Word overlap too low ({overlap_ratio:.0%}); answer may not be grounded in evidence.",
            question_satisfied=False,
            answer_shape_satisfied=shape_ok,
            metadata={
                "method": "heuristic",
                "overlap_ratio": round(overlap_ratio, 3),
                "overlap_count": len(overlap_words),
                "answer_word_count": len(answer_words),
            },
        )


class AnswerValidator:
    """
    Component responsible for verifying generated answers against retrieved evidence
    and structured knowledge at both the answer and individual claim levels.
    """

    async def validate(
        self,
        answer: str,
        evidence: Optional[RetrievalResult],
        llm: Optional[BaseLanguageModel] = None,  # retained for interface compatibility
        workspace: Optional[Any] = None,  # KnowledgeWorkspace
        evidence_items: Optional[List[Any]] = None,
        information_needs: Optional[List[Any]] = None,
        answer_contract: Optional[Any] = None,
    ) -> ValidationResult:
        start_t = time.perf_counter()
        cleaned_answer = answer.strip()

        if not cleaned_answer:
            telemetry_tracker.record_validation(time.perf_counter() - start_t)
            return ValidationResult(
                status=ValidationStatus.INSUFFICIENT_EVIDENCE,
                reason="Empty answer provided for validation.",
                question_satisfied=False,
                answer_shape_satisfied=False,
            )

        from app.knowledge.claims import is_refusal_response

        if is_refusal_response(cleaned_answer):
            telemetry_tracker.record_validation(time.perf_counter() - start_t)
            return ValidationResult(
                status=ValidationStatus.INSUFFICIENT_EVIDENCE,
                reason="System-level insufficiency response generated.",
                question_satisfied=False,
                answer_shape_satisfied=False,
                claims=[],
                unsupported_claims=[],
                metadata={"refusal": True},
            )

        # ── Evidence Fingerprinting & Cache Check (Stage 13) ─────────────────
        ev_text_parts = []
        if evidence and evidence.content:
            ev_text_parts.append(evidence.content)
        if evidence_items:
            for item in evidence_items:
                ev_text_parts.append(getattr(item, "content", str(item)))
        if workspace and getattr(workspace, "facts", None):
            for fact in workspace.facts:
                ev_text_parts.append(f"{fact.subject}-{fact.predicate}-{fact.object_value}")
        
        raw_ev_str = "|".join(ev_text_parts)
        ev_fingerprint = hashlib.sha256(raw_ev_str.encode("utf-8")).hexdigest()[:16]

        cached_val = local_cache_manager.get_validation(cleaned_answer, ev_fingerprint)
        if cached_val is not None:
            logger.debug("[VALIDATOR] Returning cached validation result.")
            telemetry_tracker.record_validation(time.perf_counter() - start_t)
            return cached_val

        if (
            (not evidence or evidence.status != RetrievalStatus.SUCCESS or not evidence.content or not evidence.content.strip())
            and (not evidence_items)
            and (not workspace or not workspace.facts)
        ):
            telemetry_tracker.record_validation(time.perf_counter() - start_t)
            return ValidationResult(
                status=ValidationStatus.INSUFFICIENT_EVIDENCE,
                reason="No valid Wikipedia evidence content available for validation.",
                question_satisfied=False,
                answer_shape_satisfied=False,
            )

        # ── 1. Claim-Level Grounding (Stage 8) ────────────────────────────────
        from app.knowledge.claims import claim_grounder
        
        extracted_claims = claim_grounder.ground_answer(
            answer=cleaned_answer,
            workspace=workspace,
            evidence_items=evidence_items or ([evidence] if evidence else None),
            information_needs=information_needs,
        )

        unsupported = [
            c for c in extracted_claims
            if c.grounding_status in (
                GroundingStatus.UNSUPPORTED,
                GroundingStatus.CONFLICTING_EVIDENCE,
                GroundingStatus.INSUFFICIENT_EVIDENCE,
            )
        ]
        partially_supported = [
            c for c in extracted_claims
            if c.grounding_status == GroundingStatus.PARTIALLY_SUPPORTED
        ]
        conflicting = [
            c for c in extracted_claims
            if c.grounding_status == GroundingStatus.CONFLICTING_EVIDENCE
        ]

        # ── 2. Check for Acknowledged / Hedged Conflicts ─────────────────────
        has_ws_conflict = bool(
            workspace and (
                getattr(workspace, "has_unresolved_conflicts", lambda: False)()
                or bool(getattr(workspace, "contradictions", None))
            )
        )
        hedged_conflict = False
        if (conflicting or has_ws_conflict) and any(
            h in cleaned_answer.lower()
            for h in ["conflict", "conflicting", "differ", "differing", "dispute", "disputed", "uncertain", "not definitive", "varies", "contested", "multiple accounts", "differing accounts"]
        ):
            hedged_conflict = True

        # When a conflict is hedged, conflicting claims describing the dispute and meta-hedging statements are accepted
        if hedged_conflict:
            unsupported = [
                c for c in extracted_claims
                if c.grounding_status in (GroundingStatus.UNSUPPORTED, GroundingStatus.INSUFFICIENT_EVIDENCE)
                and not any(
                    h in c.statement.lower()
                    for h in ["conflict", "conflicting", "differ", "differing", "dispute", "disputed", "uncertain", "not definitive", "varies", "contested", "multiple accounts", "differing accounts"]
                )
            ]

        # ── 3. Determine Overall Validation Status ────────────────────────────
        if conflicting and not hedged_conflict:
            status = ValidationStatus.CONFLICTING_EVIDENCE
            reason = f"Conflicting evidence detected in {len(conflicting)} claim(s) without appropriate qualification."
        elif unsupported:
            # If any material factual claim is unsupported, answer is UNSUPPORTED
            status = ValidationStatus.UNSUPPORTED
            reason = f"{len(unsupported)} of {len(extracted_claims)} claim(s) failed grounding against Wikipedia evidence."
        elif partially_supported and not any(c.grounding_status == GroundingStatus.SUPPORTED for c in extracted_claims):
            status = ValidationStatus.PARTIALLY_SUPPORTED
            reason = f"{len(partially_supported)} claim(s) only partially supported by retrieved evidence."
        elif not extracted_claims:
            # Fallback to word-overlap heuristic
            ev_content = evidence.content if evidence and evidence.content else ""
            res = _heuristic_validate(cleaned_answer, ev_content, answer_contract=answer_contract)
            local_cache_manager.set_validation(cleaned_answer, ev_fingerprint, res)
            telemetry_tracker.record_validation(time.perf_counter() - start_t)
            return res
        else:
            status = ValidationStatus.SUPPORTED
            reason = f"All {len(extracted_claims)} claim(s) verified and grounded in Wikipedia evidence."

        # ── 4. Shape & Question Satisfaction Validation ───────────────────────
        shape_ok, shape_reason = _check_contract_satisfaction(cleaned_answer, answer_contract)
        answer_shape_satisfied = shape_ok
        question_satisfied = (status == ValidationStatus.SUPPORTED) and answer_shape_satisfied

        if not shape_ok and status == ValidationStatus.SUPPORTED:
            status = ValidationStatus.PARTIALLY_SUPPORTED
            reason = shape_reason or "Answer text is grounded in evidence but does not satisfy the AnswerContract shape or requested relation."

        workspace_quality = None
        if workspace and hasattr(workspace, "evaluate_workspace_quality"):
            workspace_quality = workspace.evaluate_workspace_quality(information_needs, evidence_items)

        result = ValidationResult(
            status=status,
            reason=reason,
            grounding_status=status,
            question_satisfied=question_satisfied,
            answer_shape_satisfied=answer_shape_satisfied,
            claims=extracted_claims,
            unsupported_claims=[] if hedged_conflict else (unsupported + partially_supported),
            metadata={
                "claim_count": len(extracted_claims),
                "supported_count": len(extracted_claims) - len(unsupported) - len(partially_supported),
                "unsupported_count": len(unsupported),
                "partially_supported_count": len(partially_supported),
                "conflicting_count": len(conflicting),
                "hedged_conflict": hedged_conflict,
                "question_satisfied": question_satisfied,
                "answer_shape_satisfied": answer_shape_satisfied,
                "shape_reason": shape_reason,
                "knowledge_quality": workspace_quality.model_dump() if workspace_quality else None,
            },
        )

        local_cache_manager.set_validation(cleaned_answer, ev_fingerprint, result)
        telemetry_tracker.record_validation(time.perf_counter() - start_t)
        return result


answer_validator = AnswerValidator()

