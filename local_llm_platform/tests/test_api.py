import pytest
from fastapi.testclient import TestClient

from local_llm_platform.apps.gateway_api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data


class TestModelsEndpoints:
    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data

    def test_list_all_models(self, client):
        response = client.get("/v1/models/all")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_register_model(self, client):
        response = client.post("/v1/models/register", json={
            "model_id": "test-api-model",
            "display_name": "Test API Model",
            "runtime_backend": "llama_cpp",
            "model_format": "gguf",
            "status": "unloaded",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test-api-model"

    def test_unregister_model(self, client):
        client.post("/v1/models/register", json={
            "model_id": "to-delete",
            "display_name": "Delete Me",
            "runtime_backend": "llama_cpp",
            "model_format": "gguf",
        })
        response = client.delete("/v1/models/to-delete")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True

    def test_runtime_health(self, client):
        response = client.get("/v1/runtime/health")
        assert response.status_code == 200
        data = response.json()
        assert "llama_cpp" in data


class TestTrainingEndpoints:
    def test_list_training_jobs(self, client):
        response = client.get("/v1/training/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data

    def test_create_training_job(self, client):
        response = client.post("/v1/training/jobs", json={
            "base_model": "llama-3b",
            "dataset_id": "ds-123",
            "output_name": "test-finetune",
            "training_type": "lora",
            "epochs": 1,
            "execution_target": "local",
        })
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_create_and_get_job(self, client):
        create_resp = client.post("/v1/training/jobs", json={
            "base_model": "llama-3b",
            "dataset_id": "ds-123",
            "output_name": "test-finetune-2",
            "training_type": "lora",
            "epochs": 1,
            "execution_target": "local",
        })
        job_id = create_resp.json()["job_id"]

        get_resp = client.get(f"/v1/training/jobs/{job_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["job_id"] == job_id

    def test_get_nonexistent_job(self, client):
        response = client.get("/v1/training/jobs/nonexistent")
        assert response.status_code == 500


class TestDatasetsEndpoints:
    def test_list_datasets(self, client):
        response = client.get("/v1/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data

    def test_upload_dataset(self, client):
        import io
        content = b'{"messages": [{"role": "user", "content": "hi"}]}\n'
        files = {"file": ("test.jsonl", io.BytesIO(content), "application/jsonl")}
        response = client.post(
            "/v1/datasets/upload",
            files=files,
            data={"name": "test-dataset", "format": "jsonl"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-dataset"


class TestImportEndpoint:
    def test_import_invalid_zip(self, client):
        import io
        files = {"file": ("bad.zip", io.BytesIO(b"not a zip"), "application/zip")}
        response = client.post("/v1/models/import", files=files)
        assert response.status_code == 400
