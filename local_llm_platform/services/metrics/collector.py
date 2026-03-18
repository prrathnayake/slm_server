from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.metrics")


class MetricsCollector:
    """Collects and stores platform metrics."""

    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] += value

    def record(self, name: str, value: float) -> None:
        self._histograms[name].append(value)
        if len(self._histograms[name]) > 10000:
            self._histograms[name] = self._histograms[name][-5000:]

    def gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def get_summary(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time

        histogram_stats = {}
        for name, values in self._histograms.items():
            if values:
                sorted_vals = sorted(values)
                histogram_stats[name] = {
                    "count": len(values),
                    "min": sorted_vals[0],
                    "max": sorted_vals[-1],
                    "avg": sum(values) / len(values),
                    "p50": sorted_vals[len(sorted_vals) // 2],
                    "p95": sorted_vals[int(len(sorted_vals) * 0.95)],
                    "p99": sorted_vals[int(len(sorted_vals) * 0.99)],
                }

        return {
            "uptime_seconds": uptime,
            "counters": dict(self._counters),
            "histograms": histogram_stats,
            "gauges": dict(self._gauges),
        }

    def reset(self) -> None:
        self._counters.clear()
        self._histograms.clear()
        self._gauges.clear()

    def record_request(self, model_id: str, latency: float, tokens_in: int, tokens_out: int) -> None:
        self.increment("total_requests")
        self.increment(f"model:{model_id}:requests")
        self.record("request_latency", latency)
        self.record(f"model:{model_id}:latency", latency)
        self.increment("total_tokens_in", tokens_in)
        self.increment("total_tokens_out", tokens_out)

    def record_error(self, error_type: str, model_id: Optional[str] = None) -> None:
        self.increment(f"errors:{error_type}")
        if model_id:
            self.increment(f"model:{model_id}:errors")
