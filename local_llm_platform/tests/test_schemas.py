import pytest
import asyncio
from local_llm_platform.core.schemas.chat import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ToolCall,
    ToolCallFunction,
)
from local_llm_platform.core.schemas.models import (
    ModelRegistryEntry,
    ModelStatus,
    BackendType,
    ModelFormat,
    SourceType,
    Specialization,
)
from local_llm_platform.core.schemas.training import (
    TrainingConfig,
    TrainingJob,
    TrainingStatus,
    TrainingType,
    DatasetFormat,
)
from local_llm_platform.core.schemas.completion import (
    CompletionRequest,
    CompletionResponse,
)
from local_llm_platform.core.exceptions.errors import (
    PlatformError,
    ModelNotFoundError,
    ModelLoadError,
    ModelNotReadyError,
    BackendError,
    TrainingError,
    DatasetError,
    ValidationError,
    AuthenticationError,
    RateLimitError,
)


class TestChatSchemas:
    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_chat_message_with_tool_calls(self):
        tool_call = ToolCall(
            id="call_123",
            function=ToolCallFunction(name="get_weather", arguments='{"city": "NYC"}'),
        )
        msg = ChatMessage(role="assistant", tool_calls=[tool_call])
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_chat_completion_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.7,
            stream=False,
        )
        assert req.model == "test-model"
        assert len(req.messages) == 1
        assert req.temperature == 0.7

    def test_chat_completion_request_defaults(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="test")],
        )
        assert req.temperature == 1.0
        assert req.top_p == 1.0
        assert req.stream is False
        assert req.tools is None


class TestModelSchemas:
    def test_model_registry_entry(self):
        entry = ModelRegistryEntry(
            model_id="test-model",
            display_name="Test Model",
            runtime_backend=BackendType.LLAMA_CPP,
            model_format=ModelFormat.GGUF,
        )
        assert entry.model_id == "test-model"
        assert entry.status == ModelStatus.LOADING
        assert entry.provider == "local-provider"

    def test_model_status_enum(self):
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.TRAINING.value == "training"
        assert ModelStatus.FAILED.value == "failed"
        assert ModelStatus.UNLOADED.value == "unloaded"

    def test_backend_type_enum(self):
        assert BackendType.LLAMA_CPP.value == "llama_cpp"
        assert BackendType.VLLM.value == "vllm"
        assert BackendType.TGI.value == "tgi"
        assert BackendType.REMOTE_HTTP.value == "remote_http"

    def test_model_with_tags(self):
        entry = ModelRegistryEntry(
            model_id="tagged",
            display_name="Tagged Model",
            runtime_backend=BackendType.VLLM,
            model_format=ModelFormat.SAFETENSORS,
            tags=["chat", "coding", "fast"],
        )
        assert "coding" in entry.tags


class TestTrainingSchemas:
    def test_training_config(self):
        config = TrainingConfig(
            base_model="llama-3b",
            dataset_id="ds-123",
            output_name="my-finetune",
        )
        assert config.training_type == TrainingType.LORA
        assert config.epochs == 3
        assert config.lora_r == 16

    def test_training_job(self):
        config = TrainingConfig(
            base_model="llama-3b",
            dataset_id="ds-123",
            output_name="my-finetune",
        )
        job = TrainingJob(
            job_id="train-abc",
            config=config,
            status=TrainingStatus.QUEUED,
            total_epochs=3,
        )
        assert job.job_id == "train-abc"
        assert job.progress == 0.0
        assert job.execution_target == "local"

    def test_dataset_format_enum(self):
        assert DatasetFormat.JSONL.value == "jsonl"
        assert DatasetFormat.CHAT.value == "chat"
        assert DatasetFormat.INSTRUCTION.value == "instruction"


class TestCompletionSchemas:
    def test_completion_request(self):
        req = CompletionRequest(
            model="test",
            prompt="Hello world",
            max_tokens=100,
        )
        assert req.prompt == "Hello world"
        assert req.max_tokens == 100
        assert req.stream is False


class TestExceptions:
    def test_platform_error(self):
        err = PlatformError("test error", status_code=400)
        assert err.message == "test error"
        assert err.status_code == 400

    def test_model_not_found(self):
        err = ModelNotFoundError("missing-model")
        assert err.status_code == 404
        assert "missing-model" in err.message

    def test_model_load_error(self):
        err = ModelLoadError("bad-model", "out of memory")
        assert err.status_code == 500
        assert "out of memory" in err.message

    def test_model_not_ready(self):
        err = ModelNotReadyError("model-1", "loading")
        assert err.status_code == 503
        assert "loading" in err.message

    def test_backend_error(self):
        err = BackendError("vllm", "connection refused")
        assert err.status_code == 502

    def test_training_error(self):
        err = TrainingError("job-1", "CUDA error")
        assert err.status_code == 500
        assert "job-1" in err.message

    def test_dataset_error(self):
        err = DatasetError("invalid format")
        assert err.status_code == 400

    def test_validation_error(self):
        err = ValidationError("missing field")
        assert err.status_code == 400

    def test_authentication_error(self):
        err = AuthenticationError()
        assert err.status_code == 401

    def test_rate_limit_error(self):
        err = RateLimitError()
        assert err.status_code == 429
