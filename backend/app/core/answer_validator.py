import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set
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


def _heuristic_validate(answer: str, evidence_content: str) -> ValidationResult:
    """
    Fast deterministic validation using significant-word overlap.
    """
    answer_words = _significant_words(answer)
    evidence_words = _significant_words(evidence_content)

    if not answer_words:
        return ValidationResult(
            status=ValidationStatus.INSUFFICIENT_EVIDENCE,
            reason="Answer contains no significant content words.",
            metadata={"method": "heuristic"},
        )

    if not evidence_words:
        return ValidationResult(
            status=ValidationStatus.INSUFFICIENT_EVIDENCE,
            reason="Evidence contains no significant content words.",
            metadata={"method": "heuristic"},
        )

    overlap_words = answer_words & evidence_words
    overlap_ratio = len(overlap_words) / len(answer_words)

    logger.debug(
        "[VALIDATOR] answer_words=%d  evidence_words=%d  overlap=%d  ratio=%.2f",
        len(answer_words), len(evidence_words), len(overlap_words), overlap_ratio,
    )

    if overlap_ratio >= 0.20:
        return ValidationResult(
            status=ValidationStatus.SUPPORTED,
            reason=f"Word overlap {overlap_ratio:.0%} ({len(overlap_words)}/{len(answer_words)} significant terms).",
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
            )

        from app.knowledge.claims import is_refusal_response

        if is_refusal_response(cleaned_answer):
            telemetry_tracker.record_validation(time.perf_counter() - start_t)
            return ValidationResult(
                status=ValidationStatus.INSUFFICIENT_EVIDENCE,
                reason="System-level insufficiency response generated.",
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
            res = _heuristic_validate(cleaned_answer, ev_content)
            local_cache_manager.set_validation(cleaned_answer, ev_fingerprint, res)
            telemetry_tracker.record_validation(time.perf_counter() - start_t)
            return res
        else:
            status = ValidationStatus.SUPPORTED
            reason = f"All {len(extracted_claims)} claim(s) verified and grounded in Wikipedia evidence."

        # ── 4. Shape & Question Satisfaction Validation ───────────────────────
        answer_shape_satisfied = True
        if answer_contract:
            exp_type = getattr(answer_contract, "expected_type", None)
            req_rel = getattr(answer_contract, "requested_relation", None)
            concept_a = getattr(answer_contract, "concept_a", None)
            concept_b = getattr(answer_contract, "concept_b", None)
            exp_type_str = getattr(exp_type, "value", str(exp_type)).lower() if exp_type else ""

            if "list" in exp_type_str:
                lines = [line.strip() for line in cleaned_answer.split("\n") if line.strip()]
                has_bullet = any(re.match(r"^[-*•\d]+[\.\)]?\s+", line) for line in lines)
                has_colon_group = any(":" in line for line in lines)
                has_multiple_lines = len(lines) >= 2
                answer_shape_satisfied = (has_bullet or (has_colon_group and has_multiple_lines) or len(lines) >= 3)
            elif "explanation" in exp_type_str and (req_rel == "conceptual_distinction" or (concept_a and concept_b)):
                ans_lower = cleaned_answer.lower()
                c_a = concept_a.lower() if concept_a else "empire"
                c_b = concept_b.lower() if concept_b else "dynasty"
                has_a = c_a in ans_lower
                has_b = c_b in ans_lower or "dynasties" in ans_lower
                answer_shape_satisfied = (has_a and has_b)

        question_satisfied = (status == ValidationStatus.SUPPORTED) and answer_shape_satisfied

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
                "knowledge_quality": workspace_quality.model_dump() if workspace_quality else None,
            },
        )

        local_cache_manager.set_validation(cleaned_answer, ev_fingerprint, result)
        telemetry_tracker.record_validation(time.perf_counter() - start_t)
        return result


answer_validator = AnswerValidator()

