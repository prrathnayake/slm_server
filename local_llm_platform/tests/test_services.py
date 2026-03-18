import pytest
import tempfile
import os

from local_llm_platform.services.artifacts.artifact_manager import ArtifactManager
from local_llm_platform.services.metrics.collector import MetricsCollector


class TestArtifactManager:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArtifactManager(base_dir=tmpdir)
            assert am.models_dir.exists()
            assert am.adapters_dir.exists()
            assert am.gguf_dir.exists()

    def test_list_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArtifactManager(base_dir=tmpdir)
            models = am.list_artifacts("models")
            assert models == []

    def test_save_and_load_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArtifactManager(base_dir=tmpdir)
            manifest = {"model_id": "test", "format": "gguf"}
            path = am.save_manifest("test", manifest)
            assert path.exists()

            loaded = am.am.load_manifest("test")
            assert loaded["model_id"] == "test"

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArtifactManager(base_dir=tmpdir)
            assert am.delete_artifact("nonexistent") is False


class TestMetricsCollector:
    def test_increment(self):
        mc = MetricsCollector()
        mc.increment("requests")
        mc.increment("requests", 5)
        summary = mc.get_summary()
        assert summary["counters"]["requests"] == 6

    def test_record(self):
        mc = MetricsCollector()
        mc.record("latency", 0.5)
        mc.record("latency", 1.0)
        mc.record("latency", 1.5)
        summary = mc.get_summary()
        assert summary["histograms"]["latency"]["count"] == 3
        assert summary["histograms"]["latency"]["min"] == 0.5
        assert summary["histograms"]["latency"]["max"] == 1.5

    def test_gauge(self):
        mc = MetricsCollector()
        mc.gauge("gpu_memory", 0.75)
        summary = mc.get_summary()
        assert summary["gauges"]["gpu_memory"] == 0.75

    def test_record_request(self):
        mc = MetricsCollector()
        mc.record_request("model-1", 0.5, 100, 50)
        mc.record_request("model-1", 0.3, 200, 100)
        summary = mc.get_summary()
        assert summary["counters"]["total_requests"] == 2
        assert summary["counters"]["total_tokens_in"] == 300
        assert summary["counters"]["total_tokens_out"] == 150

    def test_record_error(self):
        mc = MetricsCollector()
        mc.record_error("timeout", "model-1")
        mc.record_error("timeout", "model-2")
        mc.record_error("validation")
        summary = mc.get_summary()
        assert summary["counters"]["errors:timeout"] == 2
        assert summary["counters"]["errors:validation"] == 1
        assert summary["counters"]["model:model-1:errors"] == 1

    def test_reset(self):
        mc = MetricsCollector()
        mc.increment("test")
        mc.record("latency", 1.0)
        mc.reset()
        summary = mc.get_summary()
        assert summary["counters"] == {}
        assert summary["histograms"] == {}

    def test_uptime(self):
        mc = MetricsCollector()
        summary = mc.get_summary()
        assert "uptime_seconds" in summary
        assert summary["uptime_seconds"] >= 0
