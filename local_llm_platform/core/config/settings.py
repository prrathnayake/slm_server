from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "Local LLM Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    DATABASE_URL: str = "sqlite:///./local_llm_platform.db"

    REDIS_URL: str = "redis://localhost:6379/0"

    API_KEY: Optional[str] = None
    SECRET_KEY: str = "change-me-in-production"

    HF_TOKEN: Optional[str] = None

    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATASETS_DIR: Path = BASE_DIR / "datasets"
    JOBS_DIR: Path = BASE_DIR / "jobs"
    LOGS_DIR: Path = BASE_DIR / "logs"

    DEFAULT_BACKEND: str = "llama_cpp"
    MAX_LOADED_MODELS: int = 3
    MODEL_LOAD_TIMEOUT: int = 300

    TRAINING_WORKER_CONCURRENCY: int = 1
    MAX_TRAINING_JOBS: int = 2

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_security()

    def _validate_security(self) -> None:
        import os
        if self.SECRET_KEY == "change-me-in-production" and not self.DEBUG:
            import logging
            logging.getLogger("local_llm_platform").warning(
                "SECRET_KEY is using the default insecure value. "
                "Set SECRET_KEY in your .env file for production use."
            )


settings = Settings()
