"""
ReflectionEngine — Central component for Stage 5 Self-Reflection & Answer Repair.

Evaluates:
  1. Evidence completeness
  2. Answer validation outcomes
  3. Gap identification (missing entities, unsupported claims, missing facts)
  4. Decision on next corrective action (COMPLETE, NEEDS_RESEARCH_REPAIR, GENERATION_REPAIR_REQUIRED, UNRECOVERABLE)
  5. Enforcement of repair action budgets and duplicate action prevention
"""
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from app.agent.reflection.models import ReflectionResult, ReflectionStatus
from app.agent.research.action import ResearchAction, ResearchActionType
from app.agent.research.context import ResearchContext
from app.agent.research.evidence import CompletionStatus
from app.core.answer_validator import ValidationResult, ValidationStatus
from app.core.query_understanding import IntentCategory, QueryUnderstanding

logger = logging.getLogger("nexia.agent.reflection")

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "to", "of", "in",
    "on", "at", "by", "for", "with", "from", "as", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "he", "she", "his",
    "her", "we", "us", "our", "you", "your", "i", "me", "my", "and",
    "but", "or", "not", "also", "so", "than", "more", "most", "very",
    "such", "about", "which", "who", "what", "when", "where", "how", "why",
    "compare", "difference", "differences", "between", "versus", "vs",
    "tell", "tells", "telling", "told", "ask", "asks", "asking", "asked",
    "explain", "explains", "explaining", "explained", "explanation",
    "describe", "describes", "describing", "described", "description",
    "give", "gives", "giving", "gave", "given", "provide", "provides",
    "know", "knows", "knowing", "knew", "known", "please", "detail", "details",
    "info", "information", "summarize", "summary", "summaries", "write",
    "wrote", "written", "writeup", "overview", "say", "says", "said",
    "mention", "mentions", "mentioned", "find", "finding", "search", "searching",
    "show", "showing", "lookup", "list", "lists", "listing", "out", "names",
}


def _norm(text: str) -> str:
    """Normalize text for duplicate check."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


RELATION_ALIASES: Dict[str, List[str]] = {
    "father": ["father", "parent", "son of", "born to", "mother"],
    "author": ["author", "written by", "wrote", "creator", "authored"],
    "founder": ["founder", "founded", "established", "created", "built"],
    "rulers": ["ruler", "king", "monarch", "dynasty", "succession", "sultan", "emperor", "princely state", "throne"],
    "conceptual_distinction": [
        "empire",
        "dynasty",
        "realm",
        "state",
        "ruled by",
        "consisted of",
        "rather than",
        "successive",
    ],
    "birth_date": ["born", "birth", "reign"],
    "death_date": ["died", "death", "passed away", "assassinated"],
}


class ReflectionEngine:
    """
    Inspects agent state, research evidence, and validation outcomes to decide
    whether the task is complete or requires targeted repair.
    """

    def reflect(
        self,
        *,
        original_request: str,
        resolved_request: str,
        qu: QueryUnderstanding,
        research_ctx: ResearchContext,
        generated_answer: Optional[str] = None,
        validation_result: Optional[ValidationResult] = None,
        workspace: Optional[Any] = None,
        repair_attempt: int = 0,
        max_repair_attempts: int = 2,
        executed_repair_queries: Optional[Set[str]] = None,
    ) -> ReflectionResult:
        """
        Reflect on current state and determine whether to complete, repair research,
        repair generation, or terminate safely.
        """
        executed_queries = set(executed_repair_queries or [])
        for action in research_ctx.completed_actions:
            executed_queries.add(_norm(action.query))
            if action.entity:
                executed_queries.add(_norm(action.entity))

        # ── 1. Check for Validation Success ──────────────────────────────────
        if (
            validation_result
            and validation_result.status == ValidationStatus.SUPPORTED
            and getattr(validation_result, "question_satisfied", True)
            and getattr(validation_result, "answer_shape_satisfied", True)
        ):
            logger.debug(
                "[REFLECTION] Validation passed (SUPPORTED & question satisfied). Task complete."
            )
            return ReflectionResult(
                status=ReflectionStatus.COMPLETE,
                stop_reason="Answer successfully verified against Wikipedia evidence.",
                metadata={"repair_attempt": repair_attempt},
            )

        # ── 2. Check for Validation Errors ───────────────────────────────────
        if validation_result and validation_result.status == ValidationStatus.VALIDATION_ERROR:
            logger.warning("[REFLECTION] Validator error encountered.")
            if repair_attempt < max_repair_attempts:
                return ReflectionResult(
                    status=ReflectionStatus.GENERATION_REPAIR_REQUIRED,
                    issue="Validation error occurred during answer verification.",
                    stop_reason="Retrying generation to produce verifiable answer.",
                    metadata={"repair_attempt": repair_attempt},
                )
            return ReflectionResult(
                status=ReflectionStatus.UNRECOVERABLE,
                issue="Validation error could not be resolved.",
                stop_reason="Validation error and repair budget exhausted.",
                metadata={"repair_attempt": repair_attempt},
            )

        # ── 3. Check Repair Budget Limits ────────────────────────────────────
        if repair_attempt >= max_repair_attempts:
            logger.info(
                "[REFLECTION] Max repair attempts (%d) reached.", max_repair_attempts
            )
            # If conflicts exist and cannot be resolved further, mark as persistent
            if workspace and hasattr(workspace, "conflicts"):
                for c in workspace.conflicts:
                    if getattr(c, "status", None) == "UNRESOLVED":
                        workspace.mark_persistent_conflict(c.id, "Repair budget exhausted without definitive resolution.")

            return ReflectionResult(
                status=ReflectionStatus.UNRECOVERABLE,
                issue="Answer did not pass validation within allowed repair attempts.",
                stop_reason="Repair budget exhausted.",
                metadata={"repair_attempt": repair_attempt},
            )

        # ── 3.5. Conflict Detection & Targeted Resolution (Stage 12) ──────────
        if workspace and hasattr(workspace, "conflicts") and workspace.conflicts:
            unresolved_conflicts = [c for c in workspace.conflicts if getattr(c, "status", None) == "UNRESOLVED"]
            for conf in unresolved_conflicts:
                conf_topic = conf.topic
                norm_conf_topic = _norm(conf_topic)
                if norm_conf_topic and norm_conf_topic not in executed_queries:
                    logger.info(
                        "[REFLECTION] Conflict detected on topic %r. Triggering targeted research repair.",
                        conf_topic
                    )
                    return ReflectionResult(
                        status=ReflectionStatus.NEEDS_RESEARCH_REPAIR,
                        issue=f"Conflicting evidence detected on topic: '{conf_topic}'",
                        missing_information=f"Clarifying Wikipedia evidence for conflicting values: {conf.conflicting_values}",
                        suggested_query=conf_topic,
                        corrective_action={
                            "type": ResearchActionType.WIKIPEDIA_SEARCH.value,
                            "query": conf_topic,
                            "objective": f"Targeted research to investigate and resolve conflict on: {conf_topic}",
                        },
                        metadata={
                            "repair_attempt": repair_attempt,
                            "strategy": "conflict_resolution_repair",
                            "conflict_id": conf.id,
                        },
                    )
                else:
                    # Query already executed for this conflict without full resolution
                    workspace.mark_persistent_conflict(conf.id, "Targeted research performed; conflicting accounts persist in Wikipedia.")

        # ── 4. Multi-Entity Comparison Gap Detection ──────────────────────────
        if qu.intent == IntentCategory.COMPARISON:
            missing_entity = self._detect_missing_comparison_entity(
                resolved_request, research_ctx
            )
            if missing_entity:
                norm_missing = _norm(missing_entity)
                if norm_missing not in executed_queries:
                    logger.info(
                        "[REFLECTION] Multi-entity gap detected: missing %r", missing_entity
                    )
                    return ReflectionResult(
                        status=ReflectionStatus.NEEDS_RESEARCH_REPAIR,
                        issue=f"Evidence missing for comparison entity: {missing_entity}",
                        missing_information=f"Wikipedia evidence for {missing_entity}",
                        target_entity=missing_entity,
                        suggested_query=missing_entity,
                        corrective_action={
                            "type": ResearchActionType.WIKIPEDIA_RETRIEVE.value,
                            "entity": missing_entity,
                            "query": missing_entity,
                            "objective": f"Targeted research for missing comparison subject: {missing_entity}",
                        },
                        metadata={
                            "repair_attempt": repair_attempt,
                            "strategy": "multi_entity_gap_fill",
                        },
                    )

        # ── 4.5. Claim-Level Grounding Failure Diagnosis (Stage 8) ────────────
        from app.knowledge.claims import is_refusal_response

        # Refusal statements (e.g. "I couldn't find that in the Wikipedia article.") are system responses, NOT claims
        valid_unsupported = [
            c for c in (validation_result.unsupported_claims if validation_result else [])
            if not is_refusal_response(c.statement)
        ]

        if valid_unsupported:
            first_unsupported = valid_unsupported[0]
            claim_stmt = first_unsupported.statement
            claim_entities = [
                ent for ent in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim_stmt)
                if _norm(ent) not in ("wikipedia", "article", "summary", "answer", "source")
            ]
            target_ent = claim_entities[0] if claim_entities else None
            
            target_query = target_ent or re.sub(r"[^\w\s]", "", claim_stmt[:60]).strip()
            norm_target = _norm(target_query)
            
            # Forbid searching for system terms like "wikipedia"
            if norm_target in ("wikipedia", "article", "summary", "answer", "source", ""):
                target_query = qu.entity or resolved_request
                norm_target = _norm(target_query)

            if norm_target and norm_target not in executed_queries:
                logger.info(
                    "[REFLECTION] Claim grounding failure detected on claim: %r. Triggering targeted repair: %r",
                    claim_stmt[:50], target_query
                )
                return ReflectionResult(
                    status=ReflectionStatus.NEEDS_RESEARCH_REPAIR,
                    issue=f"Claim failed grounding against current evidence: '{claim_stmt[:60]}'",
                    missing_information=f"Supporting evidence for: {claim_stmt[:60]}",
                    target_entity=target_ent,
                    suggested_query=target_query,
                    corrective_action={
                        "type": ResearchActionType.WIKIPEDIA_RETRIEVE.value if target_ent else ResearchActionType.WIKIPEDIA_SEARCH.value,
                        "entity": target_ent,
                        "query": target_query,
                        "objective": f"Targeted repair to ground claim: {claim_stmt[:60]}",
                    },
                    metadata={
                        "repair_attempt": repair_attempt,
                        "strategy": "claim_grounding_repair",
                        "claim_id": first_unsupported.id,
                    },
                )

        # ── 4.8. Check Unresolved Information Needs ───────────────────────────
        if research_ctx.information_needs:
            for need in research_ctx.information_needs:
                from app.agent.decomposition.models import NeedStatus
                if need.status != NeedStatus.SATISFIED:
                    repair_q = need.question
                    norm_rq = _norm(repair_q)
                    if norm_rq and norm_rq not in executed_queries and norm_rq not in ("wikipedia", ""):
                        logger.info("[REFLECTION] Unresolved InformationNeed: %r. Triggering targeted research repair.", repair_q)
                        return ReflectionResult(
                            status=ReflectionStatus.NEEDS_RESEARCH_REPAIR,
                            issue=f"Unresolved InformationNeed: '{need.purpose}'",
                            missing_information=repair_q,
                            suggested_query=repair_q,
                            target_entity=need.required_entities[0] if need.required_entities else None,
                            corrective_action={
                                "type": ResearchActionType.WIKIPEDIA_RETRIEVE.value if need.required_entities else ResearchActionType.WIKIPEDIA_SEARCH.value,
                                "entity": need.required_entities[0] if need.required_entities else None,
                                "query": repair_q,
                                "objective": f"Targeted research for unresolved need: {need.purpose}",
                            },
                            metadata={
                                "repair_attempt": repair_attempt,
                                "strategy": "information_need_repair",
                                "need_id": need.id,
                            },
                        )

        # ── 5. Missing Evidence / Specific Fact Gap Detection ─────────────────
        missing_fact_query = self._detect_missing_fact_gap(
            resolved_request, qu, research_ctx
        )
        if missing_fact_query:
            norm_fact_query = _norm(missing_fact_query)
            if norm_fact_query not in executed_queries:
                logger.info(
                    "[REFLECTION] Fact gap detected. Formulated targeted query: %r",
                    missing_fact_query
                )
                return ReflectionResult(
                    status=ReflectionStatus.NEEDS_RESEARCH_REPAIR,
                    issue="Current evidence does not contain specific information requested.",
                    missing_information=missing_fact_query,
                    suggested_query=missing_fact_query,
                    corrective_action={
                        "type": ResearchActionType.WIKIPEDIA_SEARCH.value,
                        "query": missing_fact_query,
                        "objective": f"Targeted search for missing details: {missing_fact_query}",
                    },
                    metadata={
                        "repair_attempt": repair_attempt,
                        "strategy": "targeted_fact_retrieval",
                    },
                )


        # ── 6. Generation Repair Detection (Sufficient Evidence Exists) ──────
        total_evidence_len = sum(len(e.content) for e in research_ctx.evidence)
        if total_evidence_len > 300 and generated_answer:
            logger.info(
                "[REFLECTION] Evidence is substantial (%d chars) but validation failed. Recommending generation repair.",
                total_evidence_len
            )
            return ReflectionResult(
                status=ReflectionStatus.GENERATION_REPAIR_REQUIRED,
                issue="Generated answer diverged from available evidence or failed grounding.",
                stop_reason="Regenerating answer strictly grounded in available evidence.",
                metadata={"repair_attempt": repair_attempt},
            )

        # ── 7. Safe Fallback / Unrecoverable ──────────────────────────────────
        logger.info(
            "[REFLECTION] No further targeted research actions could be formed."
        )
        return ReflectionResult(
            status=ReflectionStatus.UNRECOVERABLE,
            issue="Evidence is insufficient and no viable repair query is available.",
            stop_reason="No additional research targets found.",
            metadata={"repair_attempt": repair_attempt},
        )

    # ── Diagnostic Helpers ───────────────────────────────────────────────────

    def _detect_missing_comparison_entity(
        self, resolved_request: str, context: ResearchContext
    ) -> Optional[str]:
        """Check if any entity in a comparison question is missing from retrieved evidence."""
        # Extract entities from patterns like "compare X and Y"
        m = re.search(
            r"(?i)compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\s*\?|\s*$)",
            resolved_request,
        )
        if not m:
            m = re.search(
                r"(?i)differences?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*\?|\s*$)",
                resolved_request,
            )
        if not m:
            m = re.search(
                r"(?i)(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s*\?|\s*$)",
                resolved_request,
            )

        if m:
            entities = [m.group(1).strip(), m.group(2).strip()]
            existing_page_titles = [_norm(e.page_title) for e in context.evidence]
            existing_content = " ".join(e.content.lower() for e in context.evidence)

            for entity in entities:
                norm_ent = _norm(entity)
                # Check if entity is represented in page titles or prominently in evidence
                found = any(norm_ent in title or title in norm_ent for title in existing_page_titles)
                if not found and norm_ent not in existing_content:
                    return entity

        return None

    def _detect_missing_fact_gap(
        self,
        resolved_request: str,
        qu: QueryUnderstanding,
        context: ResearchContext,
    ) -> Optional[str]:
        """
        Extract specific target concepts from the query that are absent from evidence.
        Uses semantic relation aliases rather than pure verbatim token matching.
        """
        if not context.evidence:
            if qu.entity and qu.entity.strip():
                return qu.entity.strip()
            clean_words = [
                w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", resolved_request)
                if w.lower() not in _STOP_WORDS
            ]
            return " ".join(clean_words) if clean_words else resolved_request

        evidence_text = " ".join(e.content.lower() for e in context.evidence)
        contract = getattr(qu, "answer_contract", None)
        relation = (qu.attribute or (contract.requested_relation if contract else "") or "").lower()

        # If relation has known semantic aliases, check if ANY alias is present in evidence
        aliases = RELATION_ALIASES.get(relation, [])
        relation_covered = any(alias in evidence_text for alias in aliases) if aliases else False

        # If contract defines concept_a / concept_b (e.g. empire vs dynasty), check them
        concepts_covered = True
        if contract and (contract.concept_a or contract.concept_b):
            c_a = (contract.concept_a or "").lower()
            c_b = (contract.concept_b or "").lower()
            concepts_covered = (not c_a or c_a in evidence_text) and (not c_b or c_b in evidence_text or "dynast" in evidence_text)

        if relation_covered and concepts_covered:
            # Evidence has the semantic concepts covered; no lexical gap trigger needed
            return None

        entity_words = set(re.findall(r"\b[a-z]{2,}\b", (qu.entity or "").lower()))
        query_words = [
            w for w in re.findall(r"\b[a-z]{3,}\b", resolved_request.lower())
            if w not in _STOP_WORDS and w not in entity_words
        ]

        # Key topic words in query not found in current evidence
        missing_keywords = [w for w in query_words if w not in evidence_text]
        if missing_keywords and not (relation_covered or concepts_covered):
            if qu.entity:
                kw_str = " ".join(missing_keywords[:2])
                return f"{qu.entity} {kw_str}".strip()
            else:
                return " ".join(missing_keywords[:3]).strip()

        return None


# Global singleton
reflection_engine = ReflectionEngine()
