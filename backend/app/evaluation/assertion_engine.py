"""
Assertion Engine for Stage 14 Evaluation Framework.
Provides deterministic, semantic, and systemic validation checks across agent runs.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agent.capability import CapabilityResult
from app.core.conversation_manager import ConversationState
from app.evaluation.models import EvaluationScenario, ScenarioResult, ScenarioStatus

logger = logging.getLogger("nexia.evaluation.assertions")


class AssertionEngine:
    """
    Evaluates execution outcomes against scenario expectations.
    Separates assertions cleanly by functional domain: correctness, grounding, agent behavior,
    security, isolation, and performance.
    """

    def evaluate_scenario_result(
        self,
        scenario: EvaluationScenario,
        result: CapabilityResult,
        answer_text: str,
        conversation_state: Optional[ConversationState] = None,
        duration_ms: float = 0.0,
    ) -> ScenarioResult:
        """Run all applicable assertion checks for a scenario."""
        passed_assertions: List[str] = []
        failed_assertions: List[str] = []

        # 1. Status Assertion
        if scenario.expected_status:
            if result.status.value == scenario.expected_status:
                passed_assertions.append(f"Status matches expected '{scenario.expected_status}'")
            else:
                failed_assertions.append(
                    f"Status mismatch: expected '{scenario.expected_status}', got '{result.status.value}'"
                )

        # 2. Capability Assertion
        if scenario.expected_capability:
            actual_cap = result.metadata.get("capability") or result.metadata.get("selected_capability_id")
            if actual_cap == scenario.expected_capability:
                passed_assertions.append(f"Capability matches '{scenario.expected_capability}'")
            else:
                failed_assertions.append(
                    f"Capability mismatch: expected '{scenario.expected_capability}', got '{actual_cap}'"
                )

        # 3. Source Attribution Assertion
        if scenario.min_sources > 0:
            sources_count = len(result.sources)
            if sources_count >= scenario.min_sources:
                passed_assertions.append(f"Source count {sources_count} >= min {scenario.min_sources}")
            else:
                failed_assertions.append(
                    f"Insufficient sources: expected at least {scenario.min_sources}, got {sources_count}"
                )

        # 4. Required Content Terms
        ans_lower = answer_text.lower()
        for term in scenario.required_terms:
            if term.lower() in ans_lower:
                passed_assertions.append(f"Required term present: '{term}'")
            else:
                failed_assertions.append(f"Missing required term: '{term}'")

        # 5. Forbidden Terms (e.g. prompt injection leaked secrets or ungrounded facts)
        for term in scenario.forbidden_terms:
            if term.lower() not in ans_lower:
                passed_assertions.append(f"Forbidden term absent: '{term}'")
            else:
                failed_assertions.append(f"Forbidden term detected in response: '{term}'")

        # 6. Grounding / Claims Assertion
        if scenario.expected_grounding_status:
            claims_info = result.metadata.get("claims", {})
            actual_grounding = result.metadata.get("grounding_status")
            if not actual_grounding and claims_info:
                if claims_info.get("unsupported", 0) > 0:
                    actual_grounding = "UNSUPPORTED"
                elif claims_info.get("supported", 0) > 0:
                    actual_grounding = "SUPPORTED"

            if actual_grounding == scenario.expected_grounding_status or (
                scenario.expected_grounding_status == "SUPPORTED" and claims_info.get("supported", 0) > 0
            ):
                passed_assertions.append(f"Grounding status matches '{scenario.expected_grounding_status}'")
            else:
                failed_assertions.append(
                    f"Grounding status mismatch: expected '{scenario.expected_grounding_status}', got '{actual_grounding}'"
                )

        # 7. Performance Constraints
        perf_metrics = result.metadata.get("performance_metrics", {})
        if scenario.max_llm_calls is not None:
            actual_llm = perf_metrics.get("llm_call_count", 0)
            if actual_llm <= scenario.max_llm_calls:
                passed_assertions.append(f"LLM calls {actual_llm} <= limit {scenario.max_llm_calls}")
            else:
                failed_assertions.append(
                    f"LLM call budget exceeded: expected <= {scenario.max_llm_calls}, got {actual_llm}"
                )

        if scenario.max_wikipedia_calls is not None:
            actual_wiki = perf_metrics.get("wikipedia_request_count", 0)
            if actual_wiki <= scenario.max_wikipedia_calls:
                passed_assertions.append(f"Wikipedia calls {actual_wiki} <= limit {scenario.max_wikipedia_calls}")
            else:
                failed_assertions.append(
                    f"Wikipedia call budget exceeded: expected <= {scenario.max_wikipedia_calls}, got {actual_wiki}"
                )

        # 8. Provenance Trace Extraction
        provenance = {
            "sources": [s.model_dump() for s in result.sources],
            "claims": result.metadata.get("claims", {}),
            "research": result.metadata.get("research", {}),
            "computational_profile": result.metadata.get("computational_profile", {}),
        }

        status = ScenarioStatus.PASSED if not failed_assertions else ScenarioStatus.FAILED

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            name=scenario.name,
            category=scenario.category,
            status=status,
            duration_ms=round(duration_ms, 2),
            passed_assertions=passed_assertions,
            failed_assertions=failed_assertions,
            performance_metrics=perf_metrics,
            provenance_trace=provenance,
            error_message="; ".join(failed_assertions) if failed_assertions else None,
        )


assertion_engine = AssertionEngine()
