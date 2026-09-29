"""
KnowledgeWorkspace — Temporary per-run structured knowledge representation.

Holds entities, facts, relationships, claims, and contradictions extracted from
Wikipedia evidence during a single agent task.
"""
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

from app.knowledge.models import (
    Claim, Conflict, ConflictStatus, Contradiction, Entity, EntityType,
    EvidenceReference, Fact, FactStatus, GroundingStatus, KnowledgeQuality,
    Relationship, SupportLevel,
)


def _norm(text: str) -> str:
    """Normalize string for robust entity and fact matching."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


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


def _significant_terms(text: str) -> Set[str]:
    """Extract significant terms from a string for semantic matching."""
    if not text:
        return set()
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}


class KnowledgeWorkspace:
    """
    Temporary workspace holding structured knowledge extracted during an agent task.
    Maintains complete source provenance, tracks epistemic quality, and handles conflict lifecycle.
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id
        self.entities: Dict[str, Entity] = {}  # keyed by _norm(canonical_name)
        self.facts: List[Fact] = []
        self.relationships: List[Relationship] = []
        self.claims: List[Claim] = []
        self.contradictions: List[Contradiction] = []
        self.conflicts: List[Conflict] = []
        self.evidence_references: List[EvidenceReference] = []
        self.quality: Optional[KnowledgeQuality] = None

    def add_entity(
        self,
        canonical_name: str,
        entity_type: EntityType = EntityType.OTHER,
        aliases: Optional[List[str]] = None,
        evidence_ref: Optional[EvidenceReference] = None,
    ) -> Entity:
        """Add or update an entity with deduplication and alias merging."""
        norm_key = _norm(canonical_name)
        if not norm_key:
            raise ValueError("Entity canonical_name cannot be empty.")

        aliases_list = aliases or []

        if norm_key in self.entities:
            existing = self.entities[norm_key]
            # Merge aliases
            for alias in aliases_list:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            # Update entity type if more specific
            if existing.entity_type == EntityType.OTHER and entity_type != EntityType.OTHER:
                existing.entity_type = entity_type
            # Append evidence reference
            if evidence_ref:
                if not any(
                    r.page_title == evidence_ref.page_title and r.evidence_id == evidence_ref.evidence_id
                    for r in existing.evidence_refs
                ):
                    existing.evidence_refs.append(evidence_ref)
            return existing

        # Create new entity
        new_entity = Entity(
            canonical_name=canonical_name,
            entity_type=entity_type,
            aliases=aliases_list,
            evidence_refs=[evidence_ref] if evidence_ref else [],
        )
        self.entities[norm_key] = new_entity
        return new_entity

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        evidence_ref: Optional[EvidenceReference] = None,
        status: FactStatus = FactStatus.SUPPORTED,
        confidence: float = 1.0,
    ) -> Fact:
        """
        Add a structured fact. Deduplicates identical facts and detects contradictions.
        """
        norm_subj = _norm(subject)
        norm_pred = _norm(predicate)
        norm_obj = _norm(object_value)

        # 1. Check for identical existing fact
        for fact in self.facts:
            if (
                _norm(fact.subject) == norm_subj
                and _norm(fact.predicate) == norm_pred
                and _norm(fact.object_value) == norm_obj
            ):
                if evidence_ref:
                    if not any(
                        r.page_title == evidence_ref.page_title and r.evidence_id == evidence_ref.evidence_id
                        for r in fact.evidence_refs
                    ):
                        fact.evidence_refs.append(evidence_ref)
                return fact

        # 2. Check for conflicting fact (e.g. single-value predicates like birth date, death date)
        single_value_predicates = {
            "born", "birth_date", "birth date", "birth_year", "birth year",
            "died", "death_date", "death date", "death_year", "death year",
            "discovered_year", "discovered year",
        }
        if norm_pred in single_value_predicates:
            for fact in self.facts:
                if _norm(fact.subject) == norm_subj and _norm(fact.predicate) == norm_pred:
                    if _norm(fact.object_value) != norm_obj:
                        # Conflict detected!
                        fact.status = FactStatus.CONFLICTING
                        contradiction = self.add_contradiction(
                            topic=f"{subject} {predicate}",
                            claim_a=f"{subject} {predicate} {fact.object_value}",
                            claim_b=f"{subject} {predicate} {object_value}",
                            evidence_ref_a=fact.evidence_refs[0] if fact.evidence_refs else None,
                            evidence_ref_b=evidence_ref,
                            details=f"Conflicting values for {predicate}: '{fact.object_value}' vs '{object_value}'",
                        )

        # 3. Create new fact
        new_fact = Fact(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            evidence_refs=[evidence_ref] if evidence_ref else [],
            status=status,
            confidence=confidence,
        )
        self.facts.append(new_fact)
        return new_fact

    def add_relationship(
        self,
        source_entity: str,
        relation: str,
        target_entity: str,
        evidence_ref: Optional[EvidenceReference] = None,
        status: FactStatus = FactStatus.SUPPORTED,
    ) -> Relationship:
        """Add a directed relationship between entities with deduplication."""
        norm_src = _norm(source_entity)
        norm_rel = _norm(relation)
        norm_tgt = _norm(target_entity)

        for rel in self.relationships:
            if (
                _norm(rel.source_entity) == norm_src
                and _norm(rel.relation) == norm_rel
                and _norm(rel.target_entity) == norm_tgt
            ):
                if evidence_ref:
                    if not any(
                        r.page_title == evidence_ref.page_title and r.evidence_id == evidence_ref.evidence_id
                        for r in rel.evidence_refs
                    ):
                        rel.evidence_refs.append(evidence_ref)
                return rel

        new_rel = Relationship(
            source_entity=source_entity,
            relation=relation,
            target_entity=target_entity,
            evidence_refs=[evidence_ref] if evidence_ref else [],
            status=status,
        )
        self.relationships.append(new_rel)
        return new_rel

    def add_claim(
        self,
        statement: str,
        supporting_fact_ids: Optional[List[str]] = None,
        supporting_relationship_ids: Optional[List[str]] = None,
        evidence_ref: Optional[EvidenceReference] = None,
        status: FactStatus = FactStatus.SUPPORTED,
    ) -> Claim:
        """Add a reasoned claim supported by facts and relationships."""
        new_claim = Claim(
            statement=statement,
            supporting_fact_ids=supporting_fact_ids or [],
            supporting_relationship_ids=supporting_relationship_ids or [],
            evidence_refs=[evidence_ref] if evidence_ref else [],
            status=status,
        )
        self.claims.append(new_claim)
        return new_claim

    def add_contradiction(
        self,
        topic: str,
        claim_a: str,
        claim_b: str,
        evidence_ref_a: Optional[EvidenceReference] = None,
        evidence_ref_b: Optional[EvidenceReference] = None,
        details: str = "",
    ) -> Contradiction:
        """Record an explicit conflict between evidence sources and track Conflict state."""
        contra = Contradiction(
            topic=topic,
            claim_a=claim_a,
            claim_b=claim_b,
            evidence_ref_a=evidence_ref_a,
            evidence_ref_b=evidence_ref_b,
            details=details,
        )
        self.contradictions.append(contra)

        # Mirror into structured Conflict
        refs = [r for r in [evidence_ref_a, evidence_ref_b] if r]
        sources = [r.page_title for r in refs if r.page_title]
        self.add_conflict(
            topic=topic,
            conflicting_values=[claim_a, claim_b],
            evidence_refs=refs,
            sources=sources,
            status=ConflictStatus.UNRESOLVED,
            metadata={"contradiction_id": contra.id, "details": details},
        )
        return contra

    def add_conflict(
        self,
        topic: str,
        conflicting_values: List[str],
        evidence_refs: Optional[List[EvidenceReference]] = None,
        sources: Optional[List[str]] = None,
        fact_or_claim_id: Optional[str] = None,
        status: ConflictStatus = ConflictStatus.UNRESOLVED,
        resolution_notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conflict:
        """Record or update a structured Conflict in the workspace."""
        norm_t = _norm(topic)
        for c in self.conflicts:
            if _norm(c.topic) == norm_t:
                for val in conflicting_values:
                    if val not in c.conflicting_values:
                        c.conflicting_values.append(val)
                if evidence_refs:
                    for ref in evidence_refs:
                        if not any(r.evidence_id == ref.evidence_id for r in c.evidence_refs):
                            c.evidence_refs.append(ref)
                if sources:
                    for s in sources:
                        if s not in c.sources:
                            c.sources.append(s)
                return c

        sources_list = list(sources or [])
        if evidence_refs:
            for ref in evidence_refs:
                if ref.page_title and ref.page_title not in sources_list:
                    sources_list.append(ref.page_title)

        conflict = Conflict(
            topic=topic,
            fact_or_claim_id=fact_or_claim_id,
            conflicting_values=conflicting_values,
            evidence_refs=evidence_refs or [],
            sources=sources_list,
            status=status,
            resolution_notes=resolution_notes,
            metadata=metadata or {},
        )
        self.conflicts.append(conflict)
        return conflict

    def resolve_conflict(self, conflict_id: str, resolution_notes: str = "") -> bool:
        """Mark a conflict as resolved with resolution explanation."""
        for c in self.conflicts:
            if c.id == conflict_id or _norm(c.topic) == _norm(conflict_id):
                c.status = ConflictStatus.RESOLVED
                c.resolution_notes = resolution_notes
                return True
        return False

    def mark_persistent_conflict(self, conflict_id: str, notes: Optional[str] = None) -> bool:
        """Mark a conflict as persistent when repair budget is exhausted without resolution."""
        for c in self.conflicts:
            if c.id == conflict_id or _norm(c.topic) == _norm(conflict_id):
                c.status = ConflictStatus.PERSISTENT_CONFLICT
                if notes:
                    c.resolution_notes = notes
                return True
        return False

    def has_unresolved_conflicts(self) -> bool:
        """Check if any active unresolved conflicts exist."""
        return any(c.status == ConflictStatus.UNRESOLVED for c in self.conflicts)

    def evaluate_claim_quality(
        self,
        claim: Claim,
        evidence_items: Optional[List[Any]] = None,
    ) -> KnowledgeQuality:
        """
        Evaluate epistemic quality, support level, consistency, and agreement for an individual claim.
        """
        stmt_clean = claim.statement.strip()
        stmt_terms = _significant_terms(stmt_clean)

        # 1. Check for conflicting state
        matching_conflicts: List[Conflict] = []
        is_conflicting = False
        if claim.grounding_status == GroundingStatus.CONFLICTING_EVIDENCE or claim.status == "CONFLICTING":
            is_conflicting = True

        for c in self.conflicts:
            if c.status == ConflictStatus.UNRESOLVED:
                topic_terms = _significant_terms(c.topic)
                first_word = _norm(c.topic.split()[0]) if c.topic else ""
                if topic_terms and (topic_terms.issubset(stmt_terms) or (first_word and first_word in stmt_terms)):
                    is_conflicting = True
                    matching_conflicts.append(c)

        for contra in self.contradictions:
            topic_terms = _significant_terms(contra.topic)
            first_word = _norm(contra.topic.split()[0]) if contra.topic else ""
            if topic_terms and (topic_terms.issubset(stmt_terms) or (first_word and first_word in stmt_terms)):
                is_conflicting = True

        # Extract sources
        distinct_sources: Set[str] = set()
        for ref in claim.evidence_refs:
            if ref.page_title:
                distinct_sources.add(ref.page_title)
        for sref in claim.source_refs:
            if isinstance(sref, dict) and sref.get("page_title"):
                distinct_sources.add(sref["page_title"])

        # Determine support level
        if is_conflicting:
            support_level = SupportLevel.CONFLICTING
            consistency = "CONFLICTING"
        elif claim.grounding_status == GroundingStatus.UNSUPPORTED:
            support_level = SupportLevel.UNVERIFIED
            consistency = "UNKNOWN"
        elif claim.grounding_status == GroundingStatus.PARTIALLY_SUPPORTED:
            support_level = SupportLevel.PARTIALLY_SUPPORTED
            consistency = "CONSISTENT"
        elif claim.grounding_status == GroundingStatus.INSUFFICIENT_EVIDENCE:
            support_level = SupportLevel.INSUFFICIENT
            consistency = "UNKNOWN"
        elif claim.grounding_status == GroundingStatus.SUPPORTED:
            support_level = SupportLevel.WELL_SUPPORTED
            consistency = "CONSISTENT"
        else:
            support_level = SupportLevel.UNCERTAIN
            consistency = "UNKNOWN"

        # Evidence agreement count: number of distinct evidence refs supporting the claim
        agreement_count = max(len(claim.evidence_refs), len(distinct_sources))

        # Temporal context extraction
        temporal_match = re.search(r"\b(in\s+)?(1\d{3}|20\d{2})\b", stmt_clean, re.IGNORECASE)
        temporal_context = temporal_match.group(0) if temporal_match else claim.temporal_scope

        quality = KnowledgeQuality(
            support_level=support_level,
            evidence_coverage={"claim_grounded": support_level == SupportLevel.WELL_SUPPORTED},
            consistency=consistency,
            source_count=len(distinct_sources),
            evidence_agreement_count=agreement_count,
            unresolved_items=[c.topic for c in matching_conflicts],
            temporal_context=temporal_context,
            is_fresh=True,
            conflicts=matching_conflicts,
            status="EVALUATED",
        )

        claim.quality = quality
        claim.support_level = support_level
        if temporal_context and not claim.temporal_scope:
            claim.temporal_scope = temporal_context

        return quality

    def evaluate_workspace_quality(
        self,
        information_needs: Optional[List[Any]] = None,
        evidence_items: Optional[List[Any]] = None,
    ) -> KnowledgeQuality:
        """
        Evaluate overall knowledge quality across all information needs and workspace claims.
        """
        needs_list = information_needs or []
        evidence_list = evidence_items or []

        # Evaluate individual claims
        for claim in self.claims:
            self.evaluate_claim_quality(claim, evidence_list)

        # Check unresolved conflicts
        unresolved_confs = [c for c in self.conflicts if c.status == ConflictStatus.UNRESOLVED]
        has_conflicts = len(unresolved_confs) > 0 or len(self.contradictions) > 0

        # Compute information needs coverage
        satisfied_needs = 0
        unresolved_needs: List[str] = []
        for n in needs_list:
            n_status = getattr(n, "status", None)
            status_val = n_status.value if hasattr(n_status, "value") else str(n_status)
            if status_val == "SATISFIED":
                satisfied_needs += 1
            else:
                unresolved_needs.append(getattr(n, "question", str(n)))

        total_needs = len(needs_list)
        coverage_dict = {
            "total_needs": total_needs,
            "satisfied_needs": satisfied_needs,
            "unresolved_needs_count": len(unresolved_needs),
            "coverage_ratio": (satisfied_needs / total_needs) if total_needs > 0 else 1.0,
        }

        # Count distinct sources across evidence and facts
        distinct_sources: Set[str] = set()
        for ev in evidence_list:
            pt = getattr(ev, "page_title", None)
            if pt:
                distinct_sources.add(pt)
        for f in self.facts:
            for ref in f.evidence_refs:
                if ref.page_title:
                    distinct_sources.add(ref.page_title)

        # Determine overall support level
        if has_conflicts:
            overall_support = SupportLevel.CONFLICTING
            consistency = "CONFLICTING"
        elif total_needs > 0 and satisfied_needs == 0 and not evidence_list and not self.facts:
            overall_support = SupportLevel.INSUFFICIENT
            consistency = "UNKNOWN"
        elif total_needs > 0 and satisfied_needs < total_needs:
            overall_support = SupportLevel.PARTIALLY_SUPPORTED
            consistency = "CONSISTENT"
        elif any(c.support_level == SupportLevel.PARTIALLY_SUPPORTED for c in self.claims):
            overall_support = SupportLevel.PARTIALLY_SUPPORTED
            consistency = "CONSISTENT"
        elif any(c.support_level == SupportLevel.UNVERIFIED for c in self.claims):
            overall_support = SupportLevel.UNVERIFIED
            consistency = "UNKNOWN"
        elif self.facts or any(c.support_level == SupportLevel.WELL_SUPPORTED for c in self.claims) or evidence_list:
            overall_support = SupportLevel.WELL_SUPPORTED
            consistency = "CONSISTENT"
        else:
            overall_support = SupportLevel.UNCERTAIN
            consistency = "UNKNOWN"

        total_agreement = sum(c.quality.evidence_agreement_count for c in self.claims if c.quality)

        quality = KnowledgeQuality(
            support_level=overall_support,
            evidence_coverage=coverage_dict,
            consistency=consistency,
            source_count=len(distinct_sources),
            evidence_agreement_count=total_agreement,
            unresolved_items=unresolved_needs + [c.topic for c in unresolved_confs],
            conflicts=unresolved_confs,
            status="EVALUATED",
        )

        self.quality = quality
        return quality

    def find_entities(self, query: str = "") -> List[Entity]:
        """Search entities by canonical name or alias."""
        norm_q = _norm(query)
        if not norm_q:
            return list(self.entities.values())
        results = []
        for ent in self.entities.values():
            if norm_q in _norm(ent.canonical_name) or any(norm_q in _norm(a) for a in ent.aliases):
                results.append(ent)
        return results

    def find_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_value: Optional[str] = None,
    ) -> List[Fact]:
        """Filter facts by subject, predicate, or object value."""
        norm_s = _norm(subject) if subject else None
        norm_p = _norm(predicate) if predicate else None
        norm_o = _norm(object_value) if object_value else None

        results = []
        for f in self.facts:
            if norm_s and norm_s not in _norm(f.subject):
                continue
            if norm_p and norm_p not in _norm(f.predicate):
                continue
            if norm_o and norm_o not in _norm(f.object_value):
                continue
            results.append(f)
        return results

    def find_relationships(
        self,
        source: Optional[str] = None,
        relation: Optional[str] = None,
        target: Optional[str] = None,
    ) -> List[Relationship]:
        """Filter relationships by source entity, relation, or target entity."""
        norm_s = _norm(source) if source else None
        norm_r = _norm(relation) if relation else None
        norm_t = _norm(target) if target else None

        results = []
        for r in self.relationships:
            if norm_s and norm_s not in _norm(r.source_entity):
                continue
            if norm_r and norm_r not in _norm(r.relation):
                continue
            if norm_t and norm_t not in _norm(r.target_entity):
                continue
            results.append(r)
        return results

    def get_claims_for_need(self, need_id: str) -> List[Claim]:
        """Retrieve all claims associated with a specific InformationNeed ID."""
        return [c for c in self.claims if c.information_need_id == need_id]

    def get_supported_claims(self) -> List[Claim]:
        """Retrieve all claims with SUPPORTED grounding status."""
        return [c for c in self.claims if getattr(c, "grounding_status", None) == "SUPPORTED" or getattr(c, "status", None) == "SUPPORTED"]

    def get_unsupported_claims(self) -> List[Claim]:
        """Retrieve all claims that are unsupported, partially supported, or conflicting."""
        return [
            c for c in self.claims
            if getattr(c, "grounding_status", None) in ("UNSUPPORTED", "PARTIALLY_SUPPORTED", "CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE")
            or getattr(c, "status", None) in ("UNSUPPORTED", "PARTIALLY_SUPPORTED", "CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE", "CONFLICTING")
        ]

    def to_summary_dict(self) -> Dict[str, Any]:
        """Observability summary of the workspace contents."""
        supported_count = len(self.get_supported_claims())
        unsupported_count = len(self.get_unsupported_claims())
        return {
            "entity_count": len(self.entities),
            "fact_count": len(self.facts),
            "relationship_count": len(self.relationships),
            "claim_count": len(self.claims),
            "supported_claims": supported_count,
            "unsupported_claims": unsupported_count,
            "contradiction_count": len(self.contradictions),
            "conflict_count": len(self.conflicts),
            "unresolved_conflicts": len([c for c in self.conflicts if c.status == ConflictStatus.UNRESOLVED]),
            "entities": [e.canonical_name for e in self.entities.values()],
            "contradictions": [c.details for c in self.contradictions],
            "quality": self.quality.model_dump() if self.quality else None,
        }

