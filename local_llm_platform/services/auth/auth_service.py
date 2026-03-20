from __future__ import annotations

from typing import Any, Dict, Optional

from local_llm_platform.core.config.settings import settings
from local_llm_platform.core.logging.logger import get_logger
from local_llm_platform.core.security.auth import hash_api_key

logger = get_logger("services.auth")


class AuthService:
    """Manages authentication for the platform."""

    def __init__(self):
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        if settings.API_KEY:
            key_hash = hash_api_key(settings.API_KEY)
            self._api_keys[key_hash] = {
                "name": "default",
                "permissions": ["read", "write", "admin"],
                "key_hint": settings.API_KEY[:4] + "***",
            }

    def validate_key(self, api_key: str) -> bool:
        if not settings.API_KEY:
            return True
        key_hash = hash_api_key(api_key)
        return key_hash in self._api_keys

    def add_key(self, key: str, name: str, permissions: Optional[list[str]] = None) -> None:
        key_hash = hash_api_key(key)
        self._api_keys[key_hash] = {
            "name": name,
            "permissions": permissions or ["read"],
            "key_hint": key[:4] + "***",
        }
        logger.info(f"Added API key for: {name}")

    def revoke_key(self, key: str) -> bool:
        key_hash = hash_api_key(key)
        if key_hash in self._api_keys:
            del self._api_keys[key_hash]
            logger.info(f"Revoked API key")
            return True
        return False

    def list_keys(self) -> list[Dict[str, Any]]:
        return [
            {"name": info["name"], "permissions": info["permissions"], "key_hint": info.get("key_hint", "***")}
            for info in self._api_keys.values()
        ]
