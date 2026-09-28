from typing import Dict
from pydantic import BaseModel, Field
from app.core.query_understanding import QueryUnderstanding, IntentCategory
from app.core.context_manager import ContextResolution, ResolutionStatus

class ResponseMode:
    GROUNDED_KNOWLEDGE = "grounded_knowledge"
    CONVERSATIONAL = "conversational"
    UNSUPPORTED_KNOWLEDGE = "unsupported_knowledge"

class ResponsePlan(BaseModel):
    """Structured decision plan governing component execution in Nexia Core."""
    response_mode: str  # "grounded_knowledge" | "conversational" | "unsupported_knowledge"
    retrieval_required: bool
    generation_required: bool
    validation_required: bool
    attribution_required: bool
    metadata: Dict = Field(default_factory=dict)

class DecisionPolicy:
    """Component responsible for determining the execution plan for a user request."""

    def create_plan(
        self,
        qu: QueryUnderstanding,
        resolution: ContextResolution,
    ) -> ResponsePlan:
        # Determine whether factual retrieval and validation pipeline are required
        is_knowledge = (
            qu.intent in [
                IntentCategory.FACTUAL_LOOKUP,
                IntentCategory.ENTITY_INFORMATION,
                IntentCategory.EXPLANATION,
                IntentCategory.COMPARISON,
                IntentCategory.DEFINITION,
                IntentCategory.SUMMARIZATION,
            ]
            or bool(qu.entity)
            or bool(resolution.resolved_entity)
            or resolution.resolution_status == ResolutionStatus.RESOLVED
        )

        if is_knowledge:
            return ResponsePlan(
                response_mode=ResponseMode.GROUNDED_KNOWLEDGE,
                retrieval_required=True,
                generation_required=True,
                validation_required=True,
                attribution_required=True,
                metadata={"intent": qu.intent},
            )
        else:
            return ResponsePlan(
                response_mode=ResponseMode.CONVERSATIONAL,
                retrieval_required=False,
                generation_required=True,
                validation_required=False,
                attribution_required=False,
                metadata={"intent": qu.intent},
            )

decision_policy = DecisionPolicy()
