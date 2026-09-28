"""
KnowledgeWorkspace — Temporary per-run structured knowledge representation.

Holds entities, facts, relationships, claims, and contradictions extracted from
Wikipedia evidence during a single agent task.
"""
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

from app.knowledge.models import (
    Claim, Contradiction, Entity, EntityType, EvidenceReference, Fact,
    FactStatus, Relationship,
)


def _norm(text: str) -> str:
    """Normalize string for robust entity and fact matching."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


class KnowledgeWorkspace:
    """
    Temporary workspace holding structured knowledge extracted during an agent task.
    Maintains complete source provenance and prevents unverified or duplicate knowledge.
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id
        self.entities: Dict[str, Entity] = {}  # keyed by _norm(canonical_name)
        self.facts: List[Fact] = []
        self.relationships: List[Relationship] = []
        self.claims: List[Claim] = []
        self.contradictions: List[Contradiction] = []
        self.evidence_references: List[EvidenceReference] = []

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
        single_value_predicates = {"born", "birth_date", "birth date", "died", "death_date", "death date", "discovered_year"}
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
        """Record an explicit conflict between evidence sources."""
        contra = Contradiction(
            topic=topic,
            claim_a=claim_a,
            claim_b=claim_b,
            evidence_ref_a=evidence_ref_a,
            evidence_ref_b=evidence_ref_b,
            details=details,
        )
        self.contradictions.append(contra)
        return contra

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
            "entities": [e.canonical_name for e in self.entities.values()],
            "contradictions": [c.details for c in self.contradictions],
        }

