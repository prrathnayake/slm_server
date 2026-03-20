from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.metrics")


class MetricsCollector:
    """Collects and stores platform metrics."""

    MAX_HISTOGRAM_SIZE = 10000

    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, deque] = defaultdict(deque)
        self._gauges: Dict[str, float] = {}
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] += value

    def record(self, name: str, value: float) -> None:
        hist = self._histograms[name]
        hist.append(value)
        while len(hist) > self.MAX_HISTOGRAM_SIZE:
            hist.popleft()

    def gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def _percentile(self, sorted_vals: list[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        n = len(sorted_vals)
        idx = max(0, min(n - 1, int(math.ceil(n * p)) - 1))
        return sorted_vals[idx]

    def get_summary(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time

        histogram_stats = {}
        for name, hist in self._histograms.items():
            if hist:
                n = len(hist)
                total = sum(hist)
                sorted_vals = sorted(hist)
                histogram_stats[name] = {
                    "count": n,
                    "min": sorted_vals[0],
                    "max": sorted_vals[-1],
                    "avg": total / n,
                    "p50": self._percentile(sorted_vals, 0.50),
                    "p95": self._percentile(sorted_vals, 0.95),
                    "p99": self._percentile(sorted_vals, 0.99),
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
