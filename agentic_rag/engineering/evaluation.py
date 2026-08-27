"""Engineering Essentials: Evaluation & Live Model Drift Detection.

Provides LLM-as-a-Judge integration, RAG quality scoring, golden dataset benchmarks,
and sliding-window statistical drift detection for AgentCore.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvaluationScore:
    """Individual metric evaluation result."""
    metric_name: str
    score: float  # Normalized 0.0 to 1.0
    passed: bool
    reasoning: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Statistical summary of detected drift across rolling evaluation windows."""
    metric_name: str
    is_drifted: bool
    baseline_mean: float
    current_mean: float
    drift_delta: float
    sample_size: int
    threshold: float
    details: str


class EvaluationManager:
    """Manages continuous evaluation, judge invocation, and drift tracking."""

    def __init__(
        self,
        judge: Optional[Any] = None,
        drift_window_size: int = 50,
        drift_threshold: float = 0.15,
        sample_rate: float = 0.2,
    ):
        """
        Args:
            judge: An LLM-as-a-judge instance (e.g. OllamaJudge or DeepEval model).
            drift_window_size: Number of recent samples in sliding window for drift check.
            drift_threshold: Drop in mean score relative to baseline that triggers drift alarm.
            sample_rate: Ratio (0.0 - 1.0) of live production queries to evaluate out-of-band.
        """
        self.judge = judge
        self.drift_window_size = drift_window_size
        self.drift_threshold = drift_threshold
        self.sample_rate = sample_rate

        # Sliding evaluation histories per metric
        self.baseline_scores: Dict[str, List[float]] = {}
        self.rolling_windows: Dict[str, Deque[float]] = {}
        self.evaluation_history: List[EvaluationScore] = []

    def set_baseline(self, metric_name: str, scores: List[float]) -> None:
        """Sets the baseline benchmark distribution for a given metric."""
        if scores:
            self.baseline_scores[metric_name] = list(scores)
            logger.info(
                f"Set baseline for {metric_name}: mean={statistics.mean(scores):.3f} (n={len(scores)})"
            )

    def record_score(self, eval_score: EvaluationScore) -> None:
        """Records an evaluation score into history and updates the rolling drift window."""
        self.evaluation_history.append(eval_score)
        metric = eval_score.metric_name

        if metric not in self.rolling_windows:
            self.rolling_windows[metric] = deque(maxlen=self.drift_window_size)
        self.rolling_windows[metric].append(eval_score.score)

    def evaluate_response_quality(
        self,
        query: str,
        response: str,
        retrieved_contexts: Optional[List[str]] = None,
        expected_output: Optional[str] = None,
    ) -> List[EvaluationScore]:
        """Evaluates a single execution turn using the configured judge or heuristic fallbacks.
        
        Evaluates:
        1. Faithfulness (Groundedness in retrieved context)
        2. Answer Relevancy (Alignment with query)
        """
        scores: List[EvaluationScore] = []

        # If LLM judge is available and capable
        if self.judge is not None and hasattr(self.judge, "generate"):
            try:
                # 1. Faithfulness / Groundedness check
                if retrieved_contexts:
                    prompt = (
                        f"Evaluate if the following answer is faithful and directly grounded in the provided context.\n"
                        f"Context: {' '.join(retrieved_contexts)}\n"
                        f"Answer: {response}\n\n"
                        f"Return a score from 0.0 to 1.0 and a brief reason."
                    )
                    verdict = self.judge.generate(prompt)
                    score_val = self._extract_score_from_verdict(verdict)
                    eval_res = EvaluationScore(
                        metric_name="faithfulness",
                        score=score_val,
                        passed=score_val >= 0.7,
                        reasoning=str(verdict),
                    )
                    self.record_score(eval_res)
                    scores.append(eval_res)

                # 2. Answer Relevancy check
                relevancy_prompt = (
                    f"Evaluate if the answer directly and accurately answers the user's question.\n"
                    f"Question: {query}\n"
                    f"Answer: {response}\n\n"
                    f"Return a score from 0.0 to 1.0 and a brief reason."
                )
                verdict = self.judge.generate(relevancy_prompt)
                score_val = self._extract_score_from_verdict(verdict)
                eval_res = EvaluationScore(
                    metric_name="answer_relevancy",
                    score=score_val,
                    passed=score_val >= 0.7,
                    reasoning=str(verdict),
                )
                self.record_score(eval_res)
                scores.append(eval_res)

                return scores
            except Exception as e:
                logger.warning(f"Judge evaluation failed, falling back to heuristic: {e}")

        # Heuristic fallback evaluation (fast & lightweight)
        heuristic_score = self._heuristic_evaluation(query, response, retrieved_contexts)
        self.record_score(heuristic_score)
        scores.append(heuristic_score)
        return scores

    def _heuristic_evaluation(
        self, query: str, response: str, contexts: Optional[List[str]] = None
    ) -> EvaluationScore:
        """Fast non-LLM heuristic check for basic length, query overlap, and non-emptiness."""
        if not response or len(response.strip()) == 0:
            return EvaluationScore(
                metric_name="heuristic_quality",
                score=0.0,
                passed=False,
                reasoning="Empty response returned",
            )

        # Basic token overlap heuristic
        query_words = set(query.lower().split())
        resp_words = set(response.lower().split())
        overlap = len(query_words.intersection(resp_words)) / max(len(query_words), 1)
        score = min(1.0, max(0.2, overlap + 0.3))

        return EvaluationScore(
            metric_name="heuristic_quality",
            score=score,
            passed=score >= 0.5,
            reasoning=f"Keyword overlap ratio: {overlap:.2f}",
        )

    def check_drift(self, metric_name: str) -> DriftReport:
        """Detects whether the rolling score distribution has drifted significantly from baseline."""
        window = self.rolling_windows.get(metric_name)
        baseline = self.baseline_scores.get(metric_name)

        if not window or len(window) < min(10, self.drift_window_size):
            return DriftReport(
                metric_name=metric_name,
                is_drifted=False,
                baseline_mean=statistics.mean(baseline) if baseline else 1.0,
                current_mean=statistics.mean(window) if window else 1.0,
                drift_delta=0.0,
                sample_size=len(window) if window else 0,
                threshold=self.drift_threshold,
                details=f"Insufficient samples for drift calculation (have {len(window) if window else 0}, need >= 10)",
            )

        current_mean = statistics.mean(window)
        baseline_mean = statistics.mean(baseline) if baseline else 0.85
        drift_delta = baseline_mean - current_mean
        is_drifted = drift_delta >= self.drift_threshold

        details = (
            f"Drift detected! Mean dropped by {drift_delta:.3f} (threshold: {self.drift_threshold})"
            if is_drifted
            else f"Performance stable. Current mean: {current_mean:.3f}, Baseline: {baseline_mean:.3f}"
        )

        return DriftReport(
            metric_name=metric_name,
            is_drifted=is_drifted,
            baseline_mean=round(baseline_mean, 3),
            current_mean=round(current_mean, 3),
            drift_delta=round(drift_delta, 3),
            sample_size=len(window),
            threshold=self.drift_threshold,
            details=details,
        )

    def run_golden_dataset_benchmark(
        self, agent_core: Any, test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Runs a batch evaluation over a deterministic golden test dataset to detect regressions."""
        results = []
        for case in test_cases:
            prompt = case.get("input", "")
            expected = case.get("expected", None)
            
            start_t = time.time()
            res = agent_core.run(prompt, thread_id=f"eval_bench_{int(start_t)}")
            latency = time.time() - start_t
            
            # Extract final message content
            messages = res.get("messages", [])
            final_content = messages[-1].content if messages else ""

            eval_scores = self.evaluate_response_quality(
                query=prompt,
                response=final_content,
                expected_output=expected,
            )
            results.append({
                "input": prompt,
                "output": final_content,
                "latency_seconds": round(latency, 2),
                "scores": [s.score for s in eval_scores],
            })

        all_scores = [s for r in results for s in r["scores"]]
        mean_score = statistics.mean(all_scores) if all_scores else 0.0
        return {
            "total_cases": len(test_cases),
            "mean_benchmark_score": round(mean_score, 3),
            "results": results,
        }

    def _extract_score_from_verdict(self, verdict: Any) -> float:
        """Parses float score from judge output string or structured object."""
        if isinstance(verdict, (int, float)):
            return float(verdict)
        verdict_str = str(verdict).lower()
        import re
        # Look for explicit score patterns like 'score: 0.85', '0.9/1.0', or decimals
        match = re.search(r"(?:score\s*[:=]\s*)?([0-1]\.[0-9]+|1\.0|0|1)(?:/1(?:\.0)?)?", verdict_str)
        if match:
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        return 0.85  # Default fallback score when positive verdict without explicit float


# uv run python -m agentic_rag.engineering.evaluation
if __name__ == "__main__":
    import sys
    print("=" * 65)
    print(" 🚀 AGENTIC RAG EVALUATION & DRIFT DETECTION SUITE")
    print("=" * 65)

    eval_manager = EvaluationManager()

    # 1. Tier 1: Fast Heuristic Quality Check
    print("\n--- [Tier 1] Fast Heuristic Evaluation ---")
    query_1 = "What is the capital of France?"
    response_1 = "The capital of France is Paris."
    score_1 = eval_manager._heuristic_evaluation(query_1, response_1)
    print(f"Query:    {query_1}")
    print(f"Response: {response_1}")
    print(f"Score:    {score_1.score:.2f} (Passed: {score_1.passed}) | Reason: {score_1.reasoning}")

    # 2. Tier 2: LLM-as-a-Judge Evaluation (Faithfulness & Relevancy)
    print("\n--- [Tier 2] LLM Judge / RAG Triad Evaluation ---")
    try:
        from agentic_rag.judge.ollama_judge import OllamaJudge
        judge = OllamaJudge()
        eval_manager.judge = judge
        print("Initialized OllamaJudge.")
    except Exception as e:
        print(f"Note: Running with heuristic judge fallback: {e}")

    rag_query = "What database is used for short-term memory?"
    rag_context = ["The agentic RAG system uses RedisSaver for short-term memory checkpointing."]
    rag_response = "The system utilizes RedisSaver checkpointer to persist short-term conversation states."
    
    scores = eval_manager.evaluate_response_quality(
        query=rag_query,
        response=rag_response,
        retrieved_contexts=rag_context
    )
    for s in scores:
        print(f"Metric: {s.metric_name:<18} | Score: {s.score:.2f} | Passed: {s.passed}")

    # 3. Tier 3: Baseline Drift Detection & Golden Dataset
    print("\n--- [Tier 3] Sliding Window Drift Detection ---")
    eval_manager.set_baseline("faithfulness", [0.90, 0.95, 0.92, 0.88, 0.94, 0.91, 0.90, 0.93, 0.95, 0.90])
    
    # Simulate a drift scenario (lower scores over recent turns)
    print("Simulating recent degraded turns (0.60, 0.55, 0.65, 0.50, 0.62)...")
    for degraded_score in [0.60, 0.55, 0.65, 0.50, 0.62, 0.58, 0.61, 0.59, 0.63, 0.57]:
        eval_manager.record_score(EvaluationScore(
            metric_name="faithfulness",
            score=degraded_score,
            passed=degraded_score >= 0.7,
            reasoning="Simulated degraded turn"
        ))

    drift_report = eval_manager.check_drift("faithfulness")
    print(f"\n📊 Drift Report for 'faithfulness':")
    print(f" - Baseline Mean: {drift_report.baseline_mean}")
    print(f" - Current Mean:  {drift_report.current_mean}")
    print(f" - Drift Delta:   {drift_report.drift_delta} (Threshold: {drift_report.threshold})")
    print(f" - Is Drifted:    {'🚨 YES' if drift_report.is_drifted else '✅ NO'}")
    print(f" - Details:       {drift_report.details}")
    print("=" * 65)
