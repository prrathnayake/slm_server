import pytest
from fastapi.testclient import TestClient

from local_llm_platform.apps.gateway_api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestFullWorkflow:
    """Integration tests for complete user workflows."""

    def test_register_load_chat_unload_workflow(self, client):
        model_id = "integration-test-model"

        # 1. Register
        resp = client.post("/v1/models/register", json={
            "model_id": model_id,
            "display_name": "Integration Test",
            "runtime_backend": "llama_cpp",
            "model_format": "gguf",
            "status": "unloaded",
        })
        assert resp.status_code == 200
        assert resp.json()["model_id"] == model_id

        # 2. Verify in list
        resp = client.get("/v1/models/all")
        assert resp.status_code == 200
        model_ids = [m["model_id"] for m in resp.json()["data"]]
        assert model_id in model_ids

        # 3. Check it's in OpenAI format (only ready models)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        openai_ids = [m["id"] for m in resp.json()["data"]]
        assert model_id not in openai_ids  # unloaded, not ready

        # 4. Delete
        resp = client.delete(f"/v1/models/{model_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_training_job_lifecycle(self, client):
        # Create job
        resp = client.post("/v1/training/jobs", json={
            "base_model": "test-base",
            "dataset_id": "ds-test",
            "output_name": "test-output",
            "training_type": "lora",
            "epochs": 1,
            "execution_target": "local",
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # List jobs
        resp = client.get("/v1/training/jobs")
        assert resp.status_code == 200
        job_ids = [j["job_id"] for j in resp.json()["jobs"]]
        assert job_id in job_ids

        # Get job
        resp = client.get(f"/v1/training/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_id

        # Get progress
        resp = client.get(f"/v1/training/jobs/{job_id}/progress")
        assert resp.status_code == 200

        # Get logs
        resp = client.get(f"/v1/training/jobs/{job_id}/logs")
        assert resp.status_code == 200
        assert "logs" in resp.json()

    def test_dataset_upload_list(self, client):
        import io

        # Upload
        content = b'{"messages": [{"role": "user", "content": "hi"}]}\n'
        files = {"file": ("test.jsonl", io.BytesIO(content), "application/jsonl")}
        resp = client.post(
            "/v1/datasets/upload",
            files=files,
            data={"name": "test-integration", "format": "jsonl"},
        )
        assert resp.status_code == 200
        dataset_id = resp.json()["dataset_id"]

        # List
        resp = client.get("/v1/datasets")
        assert resp.status_code == 200
        ds_ids = [d["dataset_id"] for d in resp.json()["datasets"]]
        assert dataset_id in ds_ids

    def test_health_and_metrics(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "uptime_seconds" in resp.json()

    def test_runtime_health(self, client):
        resp = client.get("/v1/runtime/health")
        assert resp.status_code == 200
        assert "llama_cpp" in resp.json()
        assert "vllm" in resp.json()

    def test_error_handling(self, client):
        resp = client.get("/v1/training/jobs/nonexistent-job-id")
        assert resp.status_code == 500
