"""
Nexia Stage 6 Structured Knowledge Representation & Multi-Hop Reasoning Package.
"""
from app.knowledge.models import (
    Claim, Contradiction, Entity, EntityType, EvidenceReference, Fact,
    FactStatus, GroundingStatus, Relationship,
)
from app.knowledge.workspace import KnowledgeWorkspace
from app.knowledge.extractor import KnowledgeExtractor, knowledge_extractor
from app.knowledge.reasoning import ReasoningEngine, reasoning_engine
from app.knowledge.claims import (
    ClaimConstructor, claim_constructor,
    ClaimAligner, claim_aligner,
    ClaimGrounder, claim_grounder,
)

__all__ = [
    "Claim",
    "Contradiction",
    "Entity",
    "EntityType",
    "EvidenceReference",
    "Fact",
    "FactStatus",
    "GroundingStatus",
    "Relationship",
    "KnowledgeWorkspace",
    "KnowledgeExtractor",
    "knowledge_extractor",
    "ReasoningEngine",
    "reasoning_engine",
    "ClaimConstructor",
    "claim_constructor",
    "ClaimAligner",
    "claim_aligner",
    "ClaimGrounder",
    "claim_grounder",
]

