from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.metrics")


class MetricsCollector:
    """Collects and stores platform metrics."""

    MAX_HISTOGRAM_SIZE = 10000
    _CACHE_TTL = 0.5

    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, deque] = defaultdict(deque)
        self._hist_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._gauges: Dict[str, float] = {}
        self._start_time = time.time()
        self._cache: Dict[str, Any] = {}
        self._cache_time: float = 0.0

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] += value
        self._cache.clear()

    def record(self, name: str, value: float) -> None:
        hist = self._histograms[name]
        hist.append(value)
        if len(hist) > self.MAX_HISTOGRAM_SIZE:
            hist.popleft()
        self._hist_stats[name].clear()
        self._cache.clear()

    def gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def _compute_stats(self, hist: deque) -> Dict[str, Any]:
        n = len(hist)
        if n == 0:
            return {}
        total = sum(hist)
        sorted_vals = sorted(hist)
        idx_p50 = (n - 1) // 2
        idx_p95 = max(0, min(n - 1, int(n * 0.95)))
        idx_p99 = max(0, min(n - 1, int(n * 0.99)))
        return {
            "count": n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "avg": total / n,
            "p50": sorted_vals[idx_p50],
            "p95": sorted_vals[idx_p95],
            "p99": sorted_vals[idx_p99],
        }

    def get_summary(self) -> Dict[str, Any]:
        now = time.time()
        if self._cache and (now - self._cache_time) < self._CACHE_TTL:
            result = self._cache.copy()
            result["uptime_seconds"] = now - self._start_time
            return result

        histogram_stats = {}
        for name, hist in self._histograms.items():
            if name in self._hist_stats and self._hist_stats[name]:
                histogram_stats[name] = self._hist_stats[name]
            elif hist:
                histogram_stats[name] = self._compute_stats(hist)
                self._hist_stats[name] = histogram_stats[name]

        result = {
            "counters": dict(self._counters),
            "histograms": histogram_stats,
            "gauges": dict(self._gauges),
        }
        self._cache = result
        self._cache_time = now
        result["uptime_seconds"] = now - self._start_time
        return result

    def reset(self) -> None:
        self._counters.clear()
        self._histograms.clear()
        self._hist_stats.clear()
        self._gauges.clear()
        self._cache.clear()

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
