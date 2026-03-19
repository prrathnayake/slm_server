import pytest
import tempfile
import os

from local_llm_platform.services.versioning.versioning_service import VersioningService
from local_llm_platform.services.pool.model_pool import ModelPool
from local_llm_platform.services.config.config_manager import ConfigManager
from local_llm_platform.services.concurrency.controller import ConcurrencyController


class TestVersioningService:
    def test_create_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VersioningService(versions_dir=tmpdir)
            v = vs.create_version("model-1", "/path/to/model", {"format": "gguf"})
            assert v.version == "v1.0.0"
            assert v.model_id == "model-1"
            assert v.is_active is True

    def test_create_multiple_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VersioningService(versions_dir=tmpdir)
            v1 = vs.create_version("model-1", "/path/v1")
            v2 = vs.create_version("model-1", "/path/v2")
            assert v1.version == "v1.0.0"
            assert v2.version == "v2.0.0"
            assert v1.is_active is False
            assert v2.is_active is True

    def test_get_active_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VersioningService(versions_dir=tmpdir)
            vs.create_version("model-1", "/path/v1")
            vs.create_version("model-1", "/path/v2")
            active = vs.get_active_version("model-1")
            assert active.version == "v2.0.0"

    def test_rollback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VersioningService(versions_dir=tmpdir)
            vs.create_version("model-1", "/path/v1")
            vs.create_version("model-1", "/path/v2")
            vs.create_version("model-1", "/path/v3")

            rolled = vs.rollback("model-1", "v1.0.0")
            assert rolled.version == "v1.0.0"
            assert rolled.is_active is True

            active = vs.get_active_version("model-1")
            assert active.version == "v1.0.0"

    def test_list_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VersioningService(versions_dir=tmpdir)
            vs.create_version("model-1", "/path/v1")
            vs.create_version("model-1", "/path/v2")
            versions = vs.list_versions("model-1")
            assert len(versions) == 2

    def test_delete_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VersioningService(versions_dir=tmpdir)
            vs.create_version("model-1", "/path/v1")
            vs.create_version("model-1", "/path/v2")

            # Cannot delete active
            assert vs.delete_version("model-1", "v2.0.0") is False
            # Can delete inactive
            assert vs.delete_version("model-1", "v1.0.0") is True

    def test_version_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VersioningService(versions_dir=tmpdir)
            vs.create_version("model-1", "/path/v1")
            vs.create_version("model-1", "/path/v2")
            history = vs.get_version_history("model-1")
            assert history["total_versions"] == 2
            assert history["active_version"] == "v2.0.0"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs1 = VersioningService(versions_dir=tmpdir)
            vs1.create_version("model-1", "/path/v1")

            vs2 = VersioningService(versions_dir=tmpdir)
            active = vs2.get_active_version("model-1")
            assert active is not None
            assert active.version == "v1.0.0"


class TestModelPool:
    def test_basic_operations(self):
        pool = ModelPool(max_loaded=3)
        pool.register_loaded("m1", "llama_cpp", "/path/m1")
        pool.register_loaded("m2", "vllm", "/path/m2")
        assert len(pool._loaded_order) == 2

    def test_eviction_candidate(self):
        pool = ModelPool(max_loaded=2)
        pool.register_loaded("m1", "llama_cpp", "/path/m1")
        pool.register_loaded("m2", "vllm", "/path/m2")
        candidate = pool.get_eviction_candidate()
        assert candidate == "m1"  # oldest, not hot

    def test_hot_model_protection(self):
        pool = ModelPool(max_loaded=2)
        pool.set_hot_models(["m1"])
        pool.register_loaded("m1", "llama_cpp", "/path/m1")
        pool.register_loaded("m2", "vllm", "/path/m2")
        candidate = pool.get_eviction_candidate()
        assert candidate == "m2"  # m1 is hot

    def test_should_load(self):
        pool = ModelPool(max_loaded=2)
        pool.register_loaded("m1", "llama_cpp", "/path/m1")
        assert pool.should_load("m2") is True
        pool.register_loaded("m2", "vllm", "/path/m2")
        assert pool.should_load("m3") is True  # can evict

    def test_touch(self):
        pool = ModelPool(max_loaded=2)
        pool.register_loaded("m1", "llama_cpp", "/path/m1")
        pool.register_loaded("m2", "vllm", "/path/m2")
        pool.touch("m1")
        assert pool.get_eviction_candidate() == "m2"

    def test_load_plan(self):
        pool = ModelPool(max_loaded=2)
        plan = pool.get_load_plan("m1")
        assert plan["action"] == "load"
        assert plan["evict"] is None

    def test_status(self):
        pool = ModelPool(max_loaded=3)
        pool.set_hot_models(["m1"])
        pool.register_loaded("m1", "llama_cpp", "/path/m1")
        status = pool.get_status()
        assert status["currently_loaded"] == 1
        assert "m1" in status["hot_models"]


class TestConfigManager:
    def test_get_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            cm = ConfigManager(config_path=path)
            assert cm.get("server.port") == 8000

    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            cm = ConfigManager(config_path=path)
            cm.set("server.port", 9000)
            assert cm.get("server.port") == 9000

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            cm1 = ConfigManager(config_path=path)
            cm1.set("models.max_loaded", 5)

            cm2 = ConfigManager(config_path=path)
            assert cm2.get("models.max_loaded") == 5

    def test_get_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            cm = ConfigManager(config_path=path)
            server = cm.get_section("server")
            assert "port" in server

    def test_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            cm = ConfigManager(config_path=path)
            cm.set("server.port", 9999)
            cm.reset()
            assert cm.get("server.port") == 8000


class TestConcurrencyController:
    @pytest.mark.asyncio
    def test_acquire_release(self):
        import asyncio
        cc = ConcurrencyController(max_global=5, max_per_model=2)
        result = asyncio.get_event_loop().run_until_complete(cc.acquire("model-1"))
        assert result is True
        assert cc._total_active == 1
        cc.release("model-1")
        assert cc._total_active == 0

    def test_stats(self):
        cc = ConcurrencyController(max_global=10, max_per_model=3)
        stats = cc.get_stats()
        assert stats["max_global"] == 10
        assert stats["total_active"] == 0
        assert stats["completed"] == 0
