"""
Metrics Collector and Baseline Comparison for Stage 14 Evaluation Framework.
"""
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from app.evaluation.models import BaselineMetrics, ScenarioResult, ScenarioStatus

logger = logging.getLogger("nexia.evaluation.metrics")

_BASELINE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation_baselines.json"


class MetricsCollector:
    """
    Collects performance, latency, and resource metrics during benchmark runs,
    persisting baselines and detecting regressions.
    """

    def __init__(self, baseline_file: Optional[Path] = None):
        self.baseline_file = baseline_file or _BASELINE_FILE
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)

    def compute_summary(self, results: List[ScenarioResult]) -> Dict[str, Any]:
        """Compute aggregate benchmark metrics from a list of ScenarioResults."""
        if not results:
            return {}

        total = len(results)
        passed = sum(1 for r in results if r.status == ScenarioStatus.PASSED)
        total_duration = sum(r.duration_ms for r in results)

        total_llm_calls = 0
        total_wiki_calls = 0
        cache_hits = 0
        cache_misses = 0

        by_category: Dict[str, Dict[str, Any]] = {}

        for r in results:
            cat = r.category.value
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0, "failed": 0, "durations": []}
            by_category[cat]["total"] += 1
            if r.status == ScenarioStatus.PASSED:
                by_category[cat]["passed"] += 1
            else:
                by_category[cat]["failed"] += 1
            by_category[cat]["durations"].append(r.duration_ms)

            perf = r.performance_metrics
            total_llm_calls += perf.get("llm_call_count", 0)
            total_wiki_calls += perf.get("wikipedia_request_count", 0)
            cache_hits += perf.get("cache_hits", 0)
            cache_misses += perf.get("cache_misses", 0)

        total_lookups = cache_hits + cache_misses
        hit_rate = (cache_hits / total_lookups) if total_lookups > 0 else 0.0

        return {
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
            "average_duration_ms": round(total_duration / total, 2) if total > 0 else 0.0,
            "average_llm_calls": round(total_llm_calls / total, 2) if total > 0 else 0.0,
            "average_wikipedia_calls": round(total_wiki_calls / total, 2) if total > 0 else 0.0,
            "cache_hit_rate": round(hit_rate, 3),
            "by_category": by_category,
        }

    def save_baseline(self, stage: str, results: List[ScenarioResult]) -> BaselineMetrics:
        """Save current benchmark summary as a named baseline record."""
        summary = self.compute_summary(results)
        baseline = BaselineMetrics(
            baseline_id=f"baseline_{stage.lower()}_{int(time.time())}",
            stage=stage,
            created_at=time.time(),
            average_duration_ms=summary.get("average_duration_ms", 0.0),
            average_llm_calls=summary.get("average_llm_calls", 0.0),
            average_wikipedia_calls=summary.get("average_wikipedia_calls", 0.0),
            cache_hit_rate=summary.get("cache_hit_rate", 0.0),
            scenarios_evaluated=summary.get("total_scenarios", 0),
            metrics_by_category=summary.get("by_category", {}),
        )

        existing = self.load_all_baselines()
        existing[stage] = baseline.model_dump()

        try:
            with open(self.baseline_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
            logger.info("[METRICS] Saved performance baseline for stage '%s'", stage)
        except Exception as e:
            logger.warning("[METRICS] Failed to save baseline file: %s", e)

        return baseline

    def load_baseline(self, stage: str) -> Optional[BaselineMetrics]:
        """Load a previously saved baseline for comparison."""
        all_b = self.load_all_baselines()
        data = all_b.get(stage)
        if data:
            return BaselineMetrics(**data)
        return None

    def load_all_baselines(self) -> Dict[str, Any]:
        """Load all baseline records from disk."""
        if not self.baseline_file.exists():
            return {}
        try:
            with open(self.baseline_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def compare_against_baseline(
        self, current_results: List[ScenarioResult], baseline_stage: str = "Stage 13"
    ) -> Dict[str, Any]:
        """Compare current benchmark results against a saved baseline."""
        baseline = self.load_baseline(baseline_stage)
        current = self.compute_summary(current_results)

        if not baseline:
            return {
                "status": "NO_BASELINE",
                "message": f"No baseline found for '{baseline_stage}'",
                "current_summary": current,
            }

        duration_diff = current["average_duration_ms"] - baseline.average_duration_ms
        llm_diff = current["average_llm_calls"] - baseline.average_llm_calls
        wiki_diff = current["average_wikipedia_calls"] - baseline.average_wikipedia_calls

        return {
            "status": "COMPARED",
            "baseline_stage": baseline.stage,
            "baseline_date": baseline.created_at,
            "current_duration_ms": current["average_duration_ms"],
            "baseline_duration_ms": baseline.average_duration_ms,
            "duration_delta_ms": round(duration_diff, 2),
            "llm_calls_delta": round(llm_diff, 2),
            "wiki_calls_delta": round(wiki_diff, 2),
            "regression_detected": bool(duration_diff > 2000.0 or llm_diff > 2.0),
        }


metrics_collector = MetricsCollector()
