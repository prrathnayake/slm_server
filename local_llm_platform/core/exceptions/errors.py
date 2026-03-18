from typing import Optional


class PlatformError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelNotFoundError(PlatformError):
    """Raised when a requested model is not found in the registry."""

    def __init__(self, model_id: str):
        super().__init__(f"Model not found: {model_id}", status_code=404)
        self.model_id = model_id


class ModelLoadError(PlatformError):
    """Raised when a model fails to load into runtime."""

    def __init__(self, model_id: str, reason: Optional[str] = None):
        msg = f"Failed to load model: {model_id}"
        if reason:
            msg += f" - {reason}"
        super().__init__(msg, status_code=500)
        self.model_id = model_id


class ModelNotReadyError(PlatformError):
    """Raised when a request is made to a model that is not ready."""

    def __init__(self, model_id: str, status: str):
        super().__init__(
            f"Model '{model_id}' is not ready (status: {status})",
            status_code=503,
        )
        self.model_id = model_id
        self.model_status = status


class BackendError(PlatformError):
    """Raised when a runtime backend encounters an error."""

    def __init__(self, backend: str, message: str):
        super().__init__(f"Backend '{backend}' error: {message}", status_code=502)
        self.backend = backend


class TrainingError(PlatformError):
    """Raised when a training job fails."""

    def __init__(self, job_id: str, message: str):
        super().__init__(f"Training job '{job_id}' failed: {message}", status_code=500)
        self.job_id = job_id


class DatasetError(PlatformError):
    """Raised when there is an issue with a dataset."""

    def __init__(self, message: str):
        super().__init__(f"Dataset error: {message}", status_code=400)


class ValidationError(PlatformError):
    """Raised when input validation fails."""

    def __init__(self, message: str):
        super().__init__(f"Validation error: {message}", status_code=422)


class AuthenticationError(PlatformError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message, status_code=401)


class RateLimitError(PlatformError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)
