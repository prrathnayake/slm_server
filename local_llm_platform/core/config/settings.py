from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "Local LLM Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # Database
    DATABASE_URL: str = "sqlite:///./local_llm_platform.db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    API_KEY: Optional[str] = None
    SECRET_KEY: str = "change-me-in-production"

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATASETS_DIR: Path = BASE_DIR / "datasets"
    JOBS_DIR: Path = BASE_DIR / "jobs"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Runtime defaults
    DEFAULT_BACKEND: str = "llama_cpp"
    MAX_LOADED_MODELS: int = 3
    MODEL_LOAD_TIMEOUT: int = 300

    # Training
    TRAINING_WORKER_CONCURRENCY: int = 1
    MAX_TRAINING_JOBS: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
