"""Engineering Essentials: Observability (Tracing & Metrics).

Provides tracing, step latency tracking, and LangSmith/OpenTelemetry hooks
as described in Section 3 of Agent.md.
"""

import time
from typing import Dict, Any, List, Optional


class StepTrace:
    """Represents a traced step in the agent execution trajectory."""

    def __init__(self, step_name: str, input_data: Any):
        self.step_name = step_name
        self.input_data = input_data
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration_seconds: float = 0.0
        self.output_data: Optional[Any] = None

    def complete(self, output_data: Any):
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.output_data = output_data


class ObservabilityManager:
    """Tracks metrics, execution latency, and intermediate agent trajectories."""

    def __init__(self, enable_langsmith: bool = False):
        self.enable_langsmith = enable_langsmith
        self.traces: List[StepTrace] = []

    def start_trace(self, step_name: str, input_data: Any) -> StepTrace:
        trace = StepTrace(step_name, input_data)
        self.traces.append(trace)
        return trace

    def get_summary_metrics(self) -> Dict[str, Any]:
        return {
            "total_steps": len(self.traces),
            "total_duration": sum(t.duration_seconds for t in self.traces),
            "step_breakdown": [{"step": t.step_name, "duration_s": round(t.duration_seconds, 3)} for t in self.traces],
        }
