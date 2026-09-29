"""
Evaluation Harness and Runners for Stage 14 Evaluation Framework.
Orchestrates ScenarioRunner, RobustnessRunner, RegressionRunner, and ReportGenerator.
"""
import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, patch

from app.agent.runtime import NexiaAgentRuntime
from app.agent.capability import CapabilityResult, CapabilityStatus
from app.core.conversation_manager import conversation_manager
from app.core.llm_interface import LLMOfflineError, LLMServiceError
from app.evaluation.assertion_engine import assertion_engine
from app.evaluation.metrics import metrics_collector
from app.evaluation.models import (
    EvaluationCategory, EvaluationReport, EvaluationScenario, ScenarioResult, ScenarioStatus,
)
from app.evaluation.scenarios import (
    get_golden_scenarios, get_grounding_scenarios, get_robustness_scenarios, get_security_scenarios,
)
from app.services.wikipedia_retriever import RetrievalResult, RetrievalStatus, wikipedia_retriever
from app.services.ollama_service import ollama_service

logger = logging.getLogger("nexia.evaluation.harness")


class ScenarioRunner:
    """Executes single evaluation scenarios against the live or mocked NexiaAgentRuntime."""

    def __init__(self, runtime: Optional[NexiaAgentRuntime] = None):
        self.runtime = runtime or NexiaAgentRuntime()

    async def run_scenario(self, scenario: EvaluationScenario) -> ScenarioResult:
        """Run one scenario end-to-end and evaluate assertions."""
        cid = scenario.conversation_id or f"eval_conv_{scenario.scenario_id.lower()}"
        state = conversation_manager.get_or_create_conversation(cid)

        t_start = time.perf_counter()
        try:
            # Handle failure injection if specified
            failure_injection = scenario.metadata.get("inject_failure")
            if failure_injection == "WIKI_TIMEOUT":
                with patch.object(wikipedia_retriever, "retrieve", side_effect=asyncio.TimeoutError("Wikipedia timeout")):
                    res, _ = await self.runtime.run(
                        scenario.input_query, conversation_id=cid, user_id=scenario.user_id
                    )
            elif failure_injection == "OLLAMA_OFFLINE":
                with patch.object(ollama_service, "generate_response", side_effect=LLMOfflineError("Ollama offline")), \
                     patch.object(ollama_service, "stream_response", side_effect=LLMOfflineError("Ollama offline")):
                    res, _ = await self.runtime.run(
                        scenario.input_query, conversation_id=cid, user_id=scenario.user_id
                    )
            elif failure_injection == "MALFORMED_JSON":
                with patch.object(ollama_service, "generate_response", new_callable=AsyncMock, return_value="<<<NOT_JSON>>>"):
                    res, _ = await self.runtime.run(
                        scenario.input_query, conversation_id=cid, user_id=scenario.user_id
                    )
            else:
                res, _ = await self.runtime.run(
                    scenario.input_query, conversation_id=cid, user_id=scenario.user_id
                )

            # Consume stream to get final response
            tokens = []
            if res.stream:
                async for chunk in res.stream:
                    tokens.append(chunk)
            answer_text = "".join(tokens)
            duration_ms = (time.perf_counter() - t_start) * 1000.0

            return assertion_engine.evaluate_scenario_result(
                scenario=scenario,
                result=res,
                answer_text=answer_text,
                conversation_state=state,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            logger.warning("[EVAL] Scenario %s error: %s", scenario.scenario_id, e)
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                name=scenario.name,
                category=scenario.category,
                status=ScenarioStatus.ERROR,
                duration_ms=round(duration_ms, 2),
                error_message=str(e),
            )


class RobustnessRunner:
    """Specialized runner for fault tolerance, network anomalies, and stress scenarios."""

    def __init__(self, scenario_runner: Optional[ScenarioRunner] = None):
        self.runner = scenario_runner or ScenarioRunner()

    async def run_all_robustness(self) -> List[ScenarioResult]:
        """Execute all robustness and fault tolerance scenarios."""
        scenarios = get_robustness_scenarios()
        results = []
        for sc in scenarios:
            res = await self.runner.run_scenario(sc)
            results.append(res)
        return results


class RegressionRunner:
    """Executes Golden Scenarios and verifies system invariants against baselines."""

    def __init__(self, scenario_runner: Optional[ScenarioRunner] = None):
        self.runner = scenario_runner or ScenarioRunner()

    async def run_golden_suite(self) -> List[ScenarioResult]:
        """Run all Golden Benchmark scenarios."""
        scenarios = get_golden_scenarios()
        results = []
        for sc in scenarios:
            res = await self.runner.run_scenario(sc)
            results.append(res)
        return results


class ReportGenerator:
    """Generates structured evaluation reports with pass rates, breakdown, and baseline comparison."""

    def generate_report(
        self,
        results: List[ScenarioResult],
        baseline_stage: Optional[str] = "Stage 13",
    ) -> EvaluationReport:
        total = len(results)
        passed = sum(1 for r in results if r.status == ScenarioStatus.PASSED)
        failed = sum(1 for r in results if r.status == ScenarioStatus.FAILED)
        errors = sum(1 for r in results if r.status == ScenarioStatus.ERROR)
        pass_rate = round((passed / total) * 100.0, 1) if total > 0 else 0.0

        cat_summary: Dict[str, Dict[str, int]] = {}
        for r in results:
            cat = r.category.value
            if cat not in cat_summary:
                cat_summary[cat] = {"total": 0, "passed": 0, "failed": 0, "error": 0}
            cat_summary[cat]["total"] += 1
            if r.status == ScenarioStatus.PASSED:
                cat_summary[cat]["passed"] += 1
            elif r.status == ScenarioStatus.FAILED:
                cat_summary[cat]["failed"] += 1
            else:
                cat_summary[cat]["error"] += 1

        comparison = None
        if baseline_stage:
            comparison = metrics_collector.compare_against_baseline(results, baseline_stage=baseline_stage)

        return EvaluationReport(
            total_scenarios=total,
            passed_count=passed,
            failed_count=failed,
            error_count=errors,
            pass_rate=pass_rate,
            category_summary=cat_summary,
            scenario_results=results,
            baseline_comparison=comparison,
            system_health={"status": "HEALTHY" if errors == 0 else "DEGRADED"},
        )


class EvaluationHarness:
    """
    Central Evaluation Harness orchestrating end-to-end evaluation, benchmarks,
    robustness suites, security checks, and reporting.
    """

    def __init__(self):
        self.scenario_runner = ScenarioRunner()
        self.robustness_runner = RobustnessRunner(self.scenario_runner)
        self.regression_runner = RegressionRunner(self.scenario_runner)
        self.report_generator = ReportGenerator()

    async def run_comprehensive_evaluation(
        self, baseline_stage: str = "Stage 13"
    ) -> EvaluationReport:
        """Run all golden, grounding, security, and robustness evaluation scenarios."""
        all_scenarios = (
            get_golden_scenarios()
            + get_grounding_scenarios()
            + get_security_scenarios()
            + get_robustness_scenarios()
        )

        results: List[ScenarioResult] = []
        for sc in all_scenarios:
            res = await self.scenario_runner.run_scenario(sc)
            results.append(res)

        # Save current run as baseline if none exists
        if not metrics_collector.load_baseline("Stage 14"):
            metrics_collector.save_baseline("Stage 14", results)

        return self.report_generator.generate_report(results, baseline_stage=baseline_stage)


evaluation_harness = EvaluationHarness()
