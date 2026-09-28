"""
Models for Stage 7 Adaptive Query Decomposition & Research Strategy.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class NeedStatus(str, Enum):
    """Lifecycle status of an InformationNeed."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PARTIALLY_SATISFIED = "PARTIALLY_SATISFIED"
    SATISFIED = "SATISFIED"
    BLOCKED = "BLOCKED"
    UNRESOLVED = "UNRESOLVED"


class NeedPriority(str, Enum):
    """Execution priority of an InformationNeed."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class QueryComplexity(str, Enum):
    """Assessed complexity level of a user request."""
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"


class InformationNeed(BaseModel):
    """
    A structured unit of information required to fulfill the user's request.
    
    Fields:
        id: Unique identifier
        question: Specific sub-question or information goal
        purpose: Clear rationale of why this information is needed
        priority: HIGH, MEDIUM, LOW
        status: PENDING -> IN_PROGRESS -> SATISFIED / PARTIALLY_SATISFIED / UNRESOLVED / BLOCKED
        required_entities: List of entities required to answer this need
        supporting_evidence: List of evidence page titles or IDs satisfying this need
        dependencies: List of InformationNeed IDs that must be satisfied before this need
        depth: Decomposition hierarchy level (1 for top-level, 2 for sub-needs)
        metadata: Observability and provenance tracking
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str
    purpose: str
    priority: NeedPriority = NeedPriority.HIGH
    status: NeedStatus = NeedStatus.PENDING
    required_entities: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    depth: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)
