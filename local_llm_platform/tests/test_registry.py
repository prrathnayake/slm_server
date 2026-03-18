import pytest
import tempfile
import os
import json
from pathlib import Path

from local_llm_platform.services.registry.registry import ModelRegistry
from local_llm_platform.core.schemas.models import (
    ModelRegistryEntry,
    ModelStatus,
    BackendType,
    ModelFormat,
    SourceType,
    Specialization,
)
from local_llm_platform.core.exceptions.errors import ModelNotFoundError


@pytest.fixture
def temp_registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_registry.json")
        registry = ModelRegistry(db_path=db_path)
        yield registry


class TestModelRegistry:
    def test_register_model(self, temp_registry):
        entry = ModelRegistryEntry(
            model_id="test-model",
            display_name="Test Model",
            runtime_backend=BackendType.LLAMA_CPP,
            model_format=ModelFormat.GGUF,
        )
        result = temp_registry.register(entry)
        assert result.model_id == "test-model"
        assert result.created_at is not None

    def test_get_model(self, temp_registry):
        entry = ModelRegistryEntry(
            model_id="get-test",
            display_name="Get Test",
            runtime_backend=BackendType.VLLM,
            model_format=ModelFormat.SAFETENSORS,
        )
        temp_registry.register(entry)
        result = temp_registry.get("get-test")
        assert result.model_id == "get-test"
        assert result.display_name == "Get Test"

    def test_get_not_found(self, temp_registry):
        with pytest.raises(ModelNotFoundError):
            temp_registry.get("nonexistent")

    def test_unregister(self, temp_registry):
        entry = ModelRegistryEntry(
            model_id="to-delete",
            display_name="Delete Me",
            runtime_backend=BackendType.LLAMA_CPP,
            model_format=ModelFormat.GGUF,
        )
        temp_registry.register(entry)
        assert temp_registry.unregister("to-delete") is True
        with pytest.raises(ModelNotFoundError):
            temp_registry.get("to-delete")

    def test_unregister_not_found(self, temp_registry):
        assert temp_registry.unregister("nonexistent") is False

    def test_update_status(self, temp_registry):
        entry = ModelRegistryEntry(
            model_id="status-test",
            display_name="Status Test",
            runtime_backend=BackendType.LLAMA_CPP,
            model_format=ModelFormat.GGUF,
        )
        temp_registry.register(entry)
        updated = temp_registry.update_status("status-test", ModelStatus.READY)
        assert updated.status == ModelStatus.READY

    def test_list_models(self, temp_registry):
        for i in range(3):
            entry = ModelRegistryEntry(
                model_id=f"model-{i}",
                display_name=f"Model {i}",
                runtime_backend=BackendType.LLAMA_CPP,
                model_format=ModelFormat.GGUF,
                status=ModelStatus.READY if i < 2 else ModelStatus.FAILED,
            )
            temp_registry.register(entry)

        all_models = temp_registry.list_models()
        assert len(all_models) == 3

        ready = temp_registry.list_models(status=ModelStatus.READY)
        assert len(ready) == 2

        failed = temp_registry.list_models(status=ModelStatus.FAILED)
        assert len(failed) == 1

    def test_list_by_backend(self, temp_registry):
        temp_registry.register(ModelRegistryEntry(
            model_id="gguf-model",
            display_name="GGUF",
            runtime_backend=BackendType.LLAMA_CPP,
            model_format=ModelFormat.GGUF,
        ))
        temp_registry.register(ModelRegistryEntry(
            model_id="vllm-model",
            display_name="VLLM",
            runtime_backend=BackendType.VLLM,
            model_format=ModelFormat.SAFETENSORS,
        ))

        llama_models = temp_registry.list_models(backend="llama_cpp")
        assert len(llama_models) == 1
        assert llama_models[0].model_id == "gguf-model"

    def test_to_openai_format(self, temp_registry):
        temp_registry.register(ModelRegistryEntry(
            model_id="ready-model",
            display_name="Ready",
            runtime_backend=BackendType.LLAMA_CPP,
            model_format=ModelFormat.GGUF,
            status=ModelStatus.READY,
        ))
        temp_registry.register(ModelRegistryEntry(
            model_id="loading-model",
            display_name="Loading",
            runtime_backend=BackendType.VLLM,
            model_format=ModelFormat.SAFETENSORS,
            status=ModelStatus.LOADING,
        ))

        response = temp_registry.to_openai_format()
        model_ids = [m.id for m in response.data]
        assert "ready-model" in model_ids
        assert "loading-model" not in model_ids

    def test_search(self, temp_registry):
        temp_registry.register(ModelRegistryEntry(
            model_id="code-assist",
            display_name="Code Assistant",
            runtime_backend=BackendType.LLAMA_CPP,
            model_format=ModelFormat.GGUF,
            tags=["coding", "assistant"],
        ))
        temp_registry.register(ModelRegistryEntry(
            model_id="chat-bot",
            display_name="Chat Bot",
            runtime_backend=BackendType.VLLM,
            model_format=ModelFormat.SAFETENSORS,
            tags=["chat"],
        ))

        results = temp_registry.search("code")
        assert len(results) == 1
        assert results[0].model_id == "code-assist"

        results = temp_registry.search("chat")
        assert len(results) == 1

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "persist.json")

            r1 = ModelRegistry(db_path=db_path)
            r1.register(ModelRegistryEntry(
                model_id="persist-test",
                display_name="Persist",
                runtime_backend=BackendType.LLAMA_CPP,
                model_format=ModelFormat.GGUF,
            ))

            r2 = ModelRegistry(db_path=db_path)
            result = r2.get("persist-test")
            assert result.model_id == "persist-test"
