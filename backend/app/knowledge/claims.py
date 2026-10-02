"""
Claim Construction, Extraction, Alignment, and Grounding for Nexia Stage 8.

Establishes precise provenance and traceability:
User Goal -> Information Need -> Research Action -> Wikipedia Evidence -> Structured Knowledge -> Claim -> Generated Answer.
"""
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from app.knowledge.models import (
    Claim, Conflict, ConflictStatus, Contradiction, EvidenceReference, Fact,
    FactStatus, GroundingStatus, KnowledgeQuality, Relationship, SupportLevel,
)

logger = logging.getLogger("nexia.knowledge.claims")

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "to", "of", "in",
    "on", "at", "by", "for", "with", "from", "as", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "he", "she", "his",
    "her", "we", "us", "our", "you", "your", "i", "me", "my", "and",
    "but", "or", "not", "also", "so", "than", "more", "most", "very",
    "such", "about", "which", "who", "what", "when", "where", "how",
    "also", "known", "well", "into", "over", "after", "before", "during",
}


def _norm(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


def _significant_terms(text: str) -> Set[str]:
    """Extract significant lowercase alphabetic/numeric tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


class ClaimConstructor:
    """
    Constructs structured claims from workspace facts, relationships, and information needs.
    Maintains explicit provenance links to underlying Wikipedia evidence.
    """

    def construct_from_facts(
        self,
        facts: List[Fact],
        information_needs: Optional[List[Any]] = None,
    ) -> List[Claim]:
        """Convert structured facts into verifiable candidate claims."""
        claims: List[Claim] = []
        needs = information_needs or []

        for fact in facts:
            # Build natural statement
            stmt = f"{fact.subject} {fact.predicate} {fact.object_value}."
            
            # Find matching information need
            matched_need_id: Optional[str] = None
            norm_subj = _norm(fact.subject)
            for need in needs:
                if any(norm_subj in _norm(ent) or _norm(ent) in norm_subj for ent in getattr(need, "required_entities", [])):
                    matched_need_id = getattr(need, "id", None)
                    break
                if _norm(fact.predicate) in _norm(getattr(need, "question", "")) or _norm(fact.object_value) in _norm(getattr(need, "question", "")):
                    matched_need_id = getattr(need, "id", None)
                    break

            grounding = GroundingStatus.SUPPORTED if fact.status == FactStatus.SUPPORTED else GroundingStatus.CONFLICTING_EVIDENCE

            claim = Claim(
                statement=stmt,
                supporting_fact_ids=[fact.id],
                evidence_refs=list(fact.evidence_refs),
                source_refs=[r.model_dump() for r in fact.evidence_refs],
                grounding_status=grounding,
                information_need_id=matched_need_id,
                metadata={
                    "origin": "fact_construction",
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object_value": fact.object_value,
                },
            )
            claims.append(claim)

        return claims

    def construct_from_relationships(
        self,
        relationships: List[Relationship],
        facts: Optional[List[Fact]] = None,
        information_needs: Optional[List[Any]] = None,
    ) -> List[Claim]:
        """Convert structured entity relationships into multi-hop or relational claims."""
        claims: List[Claim] = []
        needs = information_needs or []

        for rel in relationships:
            stmt = f"{rel.source_entity} {rel.relation} {rel.target_entity}."
            
            matched_need_id: Optional[str] = None
            norm_src = _norm(rel.source_entity)
            norm_tgt = _norm(rel.target_entity)
            for need in needs:
                if any(norm_src in _norm(ent) or norm_tgt in _norm(ent) for ent in need.required_entities):
                    matched_need_id = need.id
                    break

            claim = Claim(
                statement=stmt,
                supporting_relationship_ids=[rel.id],
                evidence_refs=list(rel.evidence_refs),
                source_refs=[r.model_dump() for r in rel.evidence_refs],
                grounding_status=GroundingStatus.SUPPORTED if rel.status == FactStatus.SUPPORTED else GroundingStatus.CONFLICTING_EVIDENCE,
                information_need_id=matched_need_id,
                metadata={
                    "origin": "relationship_construction",
                    "source_entity": rel.source_entity,
                    "relation": rel.relation,
                    "target_entity": rel.target_entity,
                },
            )
            claims.append(claim)

        return claims


class ClaimAligner:
    """
    Extracts factual statements from generated answer text and segments them for verification.
    """

    def extract_statements(self, text: str) -> List[str]:
        """Split generated answer into atomic or sentence-level factual statements."""
        if not text or not text.strip():
            return []

        cleaned: List[str] = []

        # Check for markdown tables
        if "|" in text:
            for line in text.split("\n"):
                line_str = line.strip()
                if line_str.startswith("|") and line_str.endswith("|") and not re.match(r"^\|[\s\-:|]+\|$", line_str):
                    cells = [c.strip() for c in line_str.split("|")[1:-1] if c.strip()]
                    if len(cells) >= 2 and not any(h in cells[0].lower() for h in ("ruler", "king", "name", "aspect", "dimension", "header")):
                        cleaned.append(" | ".join(cells))

        if not cleaned:
            # Split on sentence boundaries and bullet points
            raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())

            for s in raw_sentences:
                s_clean = s.strip()
                # Strip markdown bullets or list numbering
                s_clean = re.sub(r"^[-*•]\s+", "", s_clean)
                s_clean = re.sub(r"^\d+\.\s+", "", s_clean).strip()
                
                # Skip empty, meta or purely conversational sentences
                if len(s_clean) < 10:
                    continue
                if re.match(r"(?i)^(hello|hi|sources?|according to wikipedia|here is|in summary)", s_clean):
                    continue
                cleaned.append(s_clean)

        return cleaned if cleaned else [text.strip()]


def is_refusal_response(text: str) -> bool:
    """Check if an answer is a system-level refusal / insufficiency template rather than a factual statement."""
    if not text:
        return False
    norm = text.strip().lower()
    refusal_patterns = [
        r"^i couldn'?t find that in the wikipedia article",
        r"^i couldn'?t find sufficient information",
        r"^i couldn'?t find information about",
        r"^i was unable to find that on wikipedia",
        r"^i do not have enough information",
        r"^i don'?t have enough information",
        r"^the retrieved wikipedia article does not contain",
    ]
    return any(re.search(p, norm) for p in refusal_patterns)


class ClaimGrounder:
    """
    Validates and grounds extracted claims against retrieved evidence, structured facts,
    relationships, and contradiction records in the KnowledgeWorkspace.
    """

    def ground_claim(
        self,
        statement: str,
        workspace: Any,  # KnowledgeWorkspace
        evidence_items: Optional[List[Any]] = None,
        information_needs: Optional[List[Any]] = None,
    ) -> Claim:
        """
        Determine the grounding status of an individual statement against the workspace and evidence.
        """
        statement_clean = statement.strip()
        stmt_terms = _significant_terms(statement_clean)
        evidence_list = evidence_items or []

        # ── 1. Check for material contradictions & conflicts in workspace ─────
        contradiction_found = False
        matching_contra: Optional[Contradiction] = None
        matching_conflict: Optional[Conflict] = None
        if workspace and getattr(workspace, "contradictions", None):
            for c in workspace.contradictions:
                topic_terms = _significant_terms(c.topic)
                first_word = _norm(c.topic.split()[0]) if c.topic else ""
                if topic_terms and (topic_terms.issubset(stmt_terms) or (first_word and first_word in stmt_terms)):
                    contradiction_found = True
                    matching_contra = c
                    break

        if not contradiction_found and workspace and getattr(workspace, "conflicts", None):
            for cf in workspace.conflicts:
                if cf.status == ConflictStatus.UNRESOLVED:
                    cf_terms = _significant_terms(cf.topic)
                    first_word = _norm(cf.topic.split()[0]) if cf.topic else ""
                    if cf_terms and (cf_terms.issubset(stmt_terms) or (first_word and first_word in stmt_terms)):
                        contradiction_found = True
                        matching_conflict = cf
                        break

        claim: Claim
        if contradiction_found:
            ev_refs = []
            s_refs = []
            c_id = ""
            details = ""
            if matching_contra:
                ev_refs = [r for r in [matching_contra.evidence_ref_a, matching_contra.evidence_ref_b] if r]
                s_refs = [r.model_dump() for r in ev_refs]
                c_id = matching_contra.id
                details = matching_contra.details
            elif matching_conflict:
                ev_refs = list(matching_conflict.evidence_refs)
                s_refs = [r.model_dump() for r in ev_refs]
                c_id = matching_conflict.id
                details = f"Conflicting values: {matching_conflict.conflicting_values}"

            claim = Claim(
                statement=statement_clean,
                grounding_status=GroundingStatus.CONFLICTING_EVIDENCE,
                evidence_refs=ev_refs,
                source_refs=s_refs,
                metadata={
                    "conflict_id": c_id,
                    "details": details,
                },
            )
        else:
            # ── 2. Collect matching evidence and facts ────────────────────────────
            matched_evidence_refs: List[EvidenceReference] = []
            matched_fact_ids: List[str] = []
            matched_rel_ids: List[str] = []

            # Check workspace facts
            if workspace and getattr(workspace, "facts", None):
                for fact in workspace.facts:
                    fact_terms = _significant_terms(f"{fact.subject} {fact.predicate} {fact.object_value}")
                    if fact_terms and len(fact_terms & stmt_terms) >= 2:
                        matched_fact_ids.append(fact.id)
                        for ref in fact.evidence_refs:
                            if not any(r.evidence_id == ref.evidence_id for r in matched_evidence_refs):
                                matched_evidence_refs.append(ref)

            # Check workspace relationships
            if workspace and getattr(workspace, "relationships", None):
                for rel in workspace.relationships:
                    rel_terms = _significant_terms(f"{rel.source_entity} {rel.relation} {rel.target_entity}")
                    if rel_terms and len(rel_terms & stmt_terms) >= 2:
                        matched_rel_ids.append(rel.id)
                        for ref in rel.evidence_refs:
                            if not any(r.evidence_id == ref.evidence_id for r in matched_evidence_refs):
                                matched_evidence_refs.append(ref)

            # Check direct text evidence items
            evidence_text_combined = " ".join(getattr(e, "content", "") for e in evidence_list).lower()
            if not matched_evidence_refs and evidence_list:
                for ev in evidence_list:
                    ev_content = getattr(ev, "content", "").lower()
                    ev_terms = _significant_terms(ev_content)
                    if stmt_terms and len(stmt_terms & ev_terms) / len(stmt_terms) >= 0.25:
                        matched_evidence_refs.append(
                            EvidenceReference(
                                page_title=getattr(ev, "page_title", "Wikipedia"),
                                page_url=getattr(ev, "page_url", "https://en.wikipedia.org"),
                                evidence_id=getattr(ev, "action_id", "ev_0"),
                            )
                        )

            # ── 3. Evaluate Grounding Precision (Dates, Entities, Overlap) ────────
            years_in_stmt = set(re.findall(r"\b(1\d{3}|20\d{2})\b", statement_clean))
            if years_in_stmt:
                unsupported_years = [y for y in years_in_stmt if y not in evidence_text_combined]
                if unsupported_years:
                    logger.debug(
                        "[CLAIM GROUNDER] Statement contains unsupported year(s) %s: %r",
                        unsupported_years, statement_clean
                    )
                    claim = Claim(
                        statement=statement_clean,
                        grounding_status=GroundingStatus.UNSUPPORTED,
                        supporting_fact_ids=matched_fact_ids,
                        evidence_refs=matched_evidence_refs,
                        source_refs=[r.model_dump() for r in matched_evidence_refs],
                        metadata={"unsupported_tokens": unsupported_years, "reason": "Specific date/number unverified"},
                    )
                    if workspace and hasattr(workspace, "evaluate_claim_quality"):
                        workspace.evaluate_claim_quality(claim, evidence_list)
                    return claim

            if not stmt_terms:
                claim = Claim(
                    statement=statement_clean,
                    grounding_status=GroundingStatus.SUPPORTED,
                    evidence_refs=matched_evidence_refs,
                    source_refs=[r.model_dump() for r in matched_evidence_refs],
                )
                if workspace and hasattr(workspace, "evaluate_claim_quality"):
                    workspace.evaluate_claim_quality(claim, evidence_list)
                return claim

            # Check term overlap with evidence
            ev_all_terms = _significant_terms(evidence_text_combined)
            overlap_terms = stmt_terms & ev_all_terms
            overlap_ratio = len(overlap_terms) / len(stmt_terms) if stmt_terms else 0.0

            # Check multi-clause / compound statements (e.g. "X and Y")
            clauses = re.split(r"\s+and\s+|\s+but\s+|;\s*", statement_clean)
            is_compound = len(clauses) > 1 and all(len(c.strip()) > 15 for c in clauses)
            if is_compound:
                clause_overlaps = []
                for c in clauses:
                    c_terms = _significant_terms(c)
                    c_ratio = len(c_terms & ev_all_terms) / len(c_terms) if c_terms else 0.0
                    clause_overlaps.append(c_ratio)
                
                if any(r >= 0.25 for r in clause_overlaps) and any(r < 0.20 for r in clause_overlaps):
                    claim = Claim(
                        statement=statement_clean,
                        grounding_status=GroundingStatus.PARTIALLY_SUPPORTED,
                        supporting_fact_ids=matched_fact_ids,
                        evidence_refs=matched_evidence_refs,
                        source_refs=[r.model_dump() for r in matched_evidence_refs],
                        metadata={"clause_overlaps": clause_overlaps, "overlap_ratio": round(overlap_ratio, 3)},
                    )
                    if workspace and hasattr(workspace, "evaluate_claim_quality"):
                        workspace.evaluate_claim_quality(claim, evidence_list)
                    return claim

            # ── 4. Grounding Status Decision ─────────────────────────────────────
            if not ev_all_terms:
                status = GroundingStatus.INSUFFICIENT_EVIDENCE
            elif overlap_ratio >= 0.25:
                status = GroundingStatus.SUPPORTED
            elif overlap_ratio >= 0.15:
                status = GroundingStatus.PARTIALLY_SUPPORTED
            else:
                status = GroundingStatus.UNSUPPORTED

            # Map to InformationNeed if provided
            matched_need_id: Optional[str] = None
            if information_needs:
                for need in information_needs:
                    if any(_norm(ent) in _norm(statement_clean) for ent in need.required_entities):
                        matched_need_id = need.id
                        break

            claim = Claim(
                statement=statement_clean,
                grounding_status=status,
                supporting_fact_ids=matched_fact_ids,
                supporting_relationship_ids=matched_rel_ids,
                evidence_refs=matched_evidence_refs,
                source_refs=[r.model_dump() for r in matched_evidence_refs],
                information_need_id=matched_need_id,
                metadata={"overlap_ratio": round(overlap_ratio, 3), "overlap_terms": list(overlap_terms)},
            )

        if workspace and hasattr(workspace, "evaluate_claim_quality"):
            workspace.evaluate_claim_quality(claim, evidence_list)
        else:
            # Fallback quality calculation
            if claim.grounding_status == GroundingStatus.CONFLICTING_EVIDENCE:
                s_level = SupportLevel.CONFLICTING
            elif claim.grounding_status == GroundingStatus.UNSUPPORTED:
                s_level = SupportLevel.UNVERIFIED
            elif claim.grounding_status == GroundingStatus.PARTIALLY_SUPPORTED:
                s_level = SupportLevel.PARTIALLY_SUPPORTED
            elif claim.grounding_status == GroundingStatus.INSUFFICIENT_EVIDENCE:
                s_level = SupportLevel.INSUFFICIENT
            elif claim.grounding_status == GroundingStatus.SUPPORTED:
                s_level = SupportLevel.WELL_SUPPORTED
            else:
                s_level = SupportLevel.UNCERTAIN

            claim.support_level = s_level
            claim.quality = KnowledgeQuality(
                support_level=s_level,
                evidence_coverage={"claim_grounded": s_level == SupportLevel.WELL_SUPPORTED},
                consistency="CONFLICTING" if s_level == SupportLevel.CONFLICTING else "CONSISTENT",
                source_count=len({r.page_title for r in claim.evidence_refs if r.page_title}),
                evidence_agreement_count=len(claim.evidence_refs),
                status="EVALUATED",
            )

        return claim

    def ground_answer(
        self,
        answer: str,
        workspace: Any,  # KnowledgeWorkspace
        evidence_items: Optional[List[Any]] = None,
        information_needs: Optional[List[Any]] = None,
    ) -> List[Claim]:
        """Extract all statements from answer and ground each individually. Refusal templates yield no claims."""
        if is_refusal_response(answer):
            logger.debug("[CLAIM GROUNDER] Answer is an insufficiency / refusal response; bypassing claim extraction.")
            return []

        aligner = ClaimAligner()
        statements = aligner.extract_statements(answer)
        claims: List[Claim] = []

        for stmt in statements:
            if is_refusal_response(stmt):
                continue
            claim = self.ground_claim(
                statement=stmt,
                workspace=workspace,
                evidence_items=evidence_items,
                information_needs=information_needs,
            )
            claims.append(claim)
            # Record in workspace if available
            if workspace and hasattr(workspace, "claims"):
                if not any(c.id == claim.id for c in workspace.claims):
                    workspace.claims.append(claim)

        return claims


claim_constructor = ClaimConstructor()
claim_aligner = ClaimAligner()
claim_grounder = ClaimGrounder()
