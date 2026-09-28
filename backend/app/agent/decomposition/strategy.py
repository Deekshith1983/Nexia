"""
ResearchStrategy — Prioritizes InformationNeeds and maps them to ResearchActions.
"""
import logging
import re
import unicodedata
import uuid
from typing import List, Optional, Set

from app.agent.decomposition.models import (
    InformationNeed, NeedPriority, NeedStatus,
)
from app.agent.research.action import (
    ResearchAction, ResearchActionStatus, ResearchActionType,
)
from app.agent.research.context import ResearchContext

logger = logging.getLogger("nexia.agent.decomposition.strategy")


def _norm(text: str) -> str:
    """Normalize text for duplicate check."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


class ResearchStrategy:
    """
    Goal-driven research strategy planner.
    Selects the highest-priority unblocked InformationNeed and generates a targeted ResearchAction.
    """

    def plan_next_action(
        self,
        needs: List[InformationNeed],
        context: ResearchContext,
    ) -> Optional[ResearchAction]:
        """
        Identify the highest priority unblocked InformationNeed that is unresolved
        and formulate a targeted Wikipedia ResearchAction.
        """
        # Collect executed queries and entities to prevent duplication
        executed_queries = {
            _norm(a.query) for a in context.completed_actions
        }
        for a in context.completed_actions:
            if a.entity:
                executed_queries.add(_norm(a.entity))

        # Filter candidate unresolved needs
        candidates: List[InformationNeed] = []
        for need in needs:
            if need.status in (NeedStatus.PENDING, NeedStatus.IN_PROGRESS, NeedStatus.PARTIALLY_SATISFIED):
                # Check dependencies
                if need.dependencies:
                    dep_needs = [n for n in needs if n.id in need.dependencies]
                    if any(d.status not in (NeedStatus.SATISFIED, NeedStatus.PARTIALLY_SATISFIED) for d in dep_needs):
                        continue
                candidates.append(need)

        if not candidates:
            return None

        # Sort by priority: HIGH (0) > MEDIUM (1) > LOW (2)
        priority_order = {
            NeedPriority.HIGH: 0,
            NeedPriority.MEDIUM: 1,
            NeedPriority.LOW: 2,
        }
        candidates.sort(key=lambda n: priority_order.get(n.priority, 1))

        # Formulate action for first non-duplicate need
        for need in candidates:
            target_entity = need.required_entities[0] if need.required_entities else None
            query_str = target_entity or need.question

            if _norm(query_str) in executed_queries:
                continue

            action_type = ResearchActionType.WIKIPEDIA_RETRIEVE if target_entity else ResearchActionType.WIKIPEDIA_SEARCH
            action = ResearchAction(
                action_id=str(uuid.uuid4())[:8],
                action_type=action_type,
                objective=need.purpose,
                query=query_str,
                entity=target_entity,
                metadata={"need_id": need.id, "priority": need.priority.value},
            )
            need.status = NeedStatus.IN_PROGRESS
            return action

        return None


research_strategy = ResearchStrategy()
