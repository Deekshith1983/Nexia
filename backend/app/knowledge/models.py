"""
Models for Stage 6 Structured Knowledge Representation & Multi-Hop Reasoning.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Categorization for structured knowledge entities."""
    PERSON = "PERSON"
    CONCEPT = "CONCEPT"
    THEORY = "THEORY"
    EVENT = "EVENT"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    WORK = "WORK"
    OTHER = "OTHER"


class FactStatus(str, Enum):
    """Grounding status of a fact, relation, or claim."""
    SUPPORTED = "SUPPORTED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTING = "CONFLICTING"


class GroundingStatus(str, Enum):
    """Grounding status of a claim against Wikipedia evidence and structured knowledge."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class SupportLevel(str, Enum):
    """Qualitative epistemic support level for claims and knowledge."""
    WELL_SUPPORTED = "WELL_SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT = "INSUFFICIENT"
    UNVERIFIED = "UNVERIFIED"


class ConflictStatus(str, Enum):
    """Lifecycle status of a factual conflict."""
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"
    PERSISTENT_CONFLICT = "PERSISTENT_CONFLICT"


class EvidenceReference(BaseModel):
    """Provenance tracking connecting structured knowledge to source Wikipedia evidence."""
    page_title: str
    page_url: str
    evidence_id: str
    snippet: Optional[str] = None


class Conflict(BaseModel):
    """Structured representation of conflicting evidence values for a topic, fact, or claim."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str
    fact_or_claim_id: Optional[str] = None
    conflicting_values: List[str] = Field(default_factory=list)
    evidence_refs: List[EvidenceReference] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    status: ConflictStatus = ConflictStatus.UNRESOLVED
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeQuality(BaseModel):
    """
    Comprehensive structured knowledge quality assessment.
    Represents epistemic support, evidence coverage, consistency, and agreement without fake probabilities.
    """
    support_level: SupportLevel = SupportLevel.UNVERIFIED
    evidence_coverage: Dict[str, Any] = Field(default_factory=dict)
    consistency: str = "CONSISTENT"  # "CONSISTENT", "CONFLICTING", "UNKNOWN"
    source_count: int = 0
    evidence_agreement_count: int = 0
    unresolved_items: List[str] = Field(default_factory=list)
    temporal_context: Optional[str] = None
    is_fresh: bool = True
    conflicts: List[Conflict] = Field(default_factory=list)
    status: str = "PENDING"


class Entity(BaseModel):
    """Structured entity derived from Wikipedia evidence."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    canonical_name: str
    entity_type: EntityType = EntityType.OTHER
    aliases: List[str] = Field(default_factory=list)
    evidence_refs: List[EvidenceReference] = Field(default_factory=list)


class Fact(BaseModel):
    """Structured subject-predicate-object fact with source provenance."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    subject: str
    predicate: str
    object_value: str
    evidence_refs: List[EvidenceReference] = Field(default_factory=list)
    status: FactStatus = FactStatus.SUPPORTED
    confidence: float = 1.0
    temporal_scope: Optional[str] = None


class Relationship(BaseModel):
    """Directed relationship between two entities supported by evidence."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_entity: str
    relation: str
    target_entity: str
    evidence_refs: List[EvidenceReference] = Field(default_factory=list)
    status: FactStatus = FactStatus.SUPPORTED


class Claim(BaseModel):
    """A reasoning unit or factual assertion synthesized from structured facts, relationships, or generated text."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    statement: str
    supporting_fact_ids: List[str] = Field(default_factory=list)
    supporting_relationship_ids: List[str] = Field(default_factory=list)
    evidence_refs: List[EvidenceReference] = Field(default_factory=list)
    source_refs: List[Dict[str, Any]] = Field(default_factory=list)
    grounding_status: GroundingStatus = GroundingStatus.SUPPORTED
    status: Optional[str] = None  # retained for backward compatibility with Stage 6
    support_level: Optional[SupportLevel] = None
    quality: Optional[KnowledgeQuality] = None
    temporal_scope: Optional[str] = None
    information_need_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.status and not self.grounding_status:
            try:
                self.grounding_status = GroundingStatus(self.status)
            except ValueError:
                self.grounding_status = GroundingStatus.SUPPORTED
        elif self.grounding_status and not self.status:
            self.status = self.grounding_status.value

        if self.quality and not self.support_level:
            self.support_level = self.quality.support_level


class Contradiction(BaseModel):
    """Explicit representation of conflicting factual claims in retrieved evidence."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str
    claim_a: str
    claim_b: str
    evidence_ref_a: Optional[EvidenceReference] = None
    evidence_ref_b: Optional[EvidenceReference] = None
    details: str

