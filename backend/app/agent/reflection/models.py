"""
Reflection Models for Nexia Stage 5 Self-Reflection & Answer Repair.
"""
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ReflectionStatus(str, Enum):
    """Lifecycle status of a reflection assessment."""
    COMPLETE = "COMPLETE"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"
    NEEDS_RESEARCH_REPAIR = "NEEDS_RESEARCH_REPAIR"
    GENERATION_REPAIR_REQUIRED = "GENERATION_REPAIR_REQUIRED"
    UNRECOVERABLE = "UNRECOVERABLE"
    CANCELLED = "CANCELLED"


class ReflectionResult(BaseModel):
    """
    Structured outcome of the ReflectionEngine's assessment.
    
    Fields:
        status: High-level determination (COMPLETE, NEEDS_RESEARCH_REPAIR, etc.)
        issue: Diagnosis of what is inadequate (if not COMPLETE)
        missing_information: Specific entity or factual gap identified
        target_entity: Identified entity name for targeted repair
        suggested_query: Recommended Wikipedia search/retrieval query
        corrective_action: Structured dictionary representing planned repair action
        stop_reason: Explanation if reflection terminates the task
        metadata: Observability payload (attempt counts, strategy, etc.)
    """
    status: ReflectionStatus
    issue: Optional[str] = None
    missing_information: Optional[str] = None
    target_entity: Optional[str] = None
    suggested_query: Optional[str] = None
    corrective_action: Optional[Dict[str, Any]] = None
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
