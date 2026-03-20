import hashlib
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from local_llm_platform.core.config.settings import settings

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def generate_api_key() -> str:
    return f"llp-{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key_plain(key1: str, key2: str) -> bool:
    return secrets.compare_digest(key1.encode(), key2.encode())


async def verify_api_key(
    authorization: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    if not settings.API_KEY:
        return None

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")

    token = authorization.replace("Bearer ", "").strip()
    if not verify_api_key_plain(token, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return token
