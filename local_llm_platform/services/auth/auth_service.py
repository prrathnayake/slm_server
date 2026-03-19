from __future__ import annotations

from typing import Any, Dict

from local_llm_platform.core.config.settings import settings
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.auth")


class AuthService:
    """Manages authentication for the platform."""

    def __init__(self):
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        if settings.API_KEY:
            self._api_keys[settings.API_KEY] = {
                "name": "default",
                "permissions": ["read", "write", "admin"],
            }

    def validate_key(self, api_key: str) -> bool:
        if not settings.API_KEY:
            return True
        return api_key in self._api_keys

    def add_key(self, key: str, name: str, permissions: list[str] = None) -> None:
        self._api_keys[key] = {
            "name": name,
            "permissions": permissions or ["read"],
        }
        logger.info(f"Added API key for: {name}")

    def revoke_key(self, key: str) -> bool:
        if key in self._api_keys:
            del self._api_keys[key]
            logger.info(f"Revoked API key")
            return True
        return False

    def list_keys(self) -> list[Dict[str, Any]]:
        return [
            {"name": info["name"], "permissions": info["permissions"]}
            for info in self._api_keys.values()
        ]
