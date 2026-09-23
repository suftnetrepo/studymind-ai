"""
API key authentication for v1 platform integrations (e.g. Learnify).

Keys look like `sm_live_<64 hex chars>` and are sent as `Authorization: Bearer <key>`.
Only the SHA-256 hash is stored; the full key is shown once on creation.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_db
from app.db.models import ApiKey

KEY_PREFIX = "sm_live_"
# "sm_live_" + 4 hex chars — enough to tell keys apart in the admin list
DISPLAY_PREFIX_LEN = 12

bearer_scheme = HTTPBearer(auto_error=False)


def generate_api_key() -> tuple[str, str, str]:
    """Returns (full_key, key_hash, key_prefix)."""
    full_key   = f"{KEY_PREFIX}{secrets.token_hex(32)}"
    key_hash   = hash_api_key(full_key)
    key_prefix = full_key[:DISPLAY_PREFIX_LEN]
    return full_key, key_hash, key_prefix


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def get_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> ApiKey:
    """
    FastAPI dependency — validates the Bearer token as a platform API key.
    Use instead of require_auth for /api/v1 platform endpoints.
    """
    unauthorized = lambda detail: HTTPException(  # noqa: E731
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not credentials or not credentials.credentials:
        raise unauthorized("API key required")

    key = credentials.credentials
    if not key.startswith(KEY_PREFIX):
        raise unauthorized("Invalid API key")

    result = await db.execute(
        select(ApiKey).where(
            ApiKey.key_hash == hash_api_key(key),
            ApiKey.is_active == True,  # noqa: E712
        )
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise unauthorized("Invalid API key")

    now = datetime.now(timezone.utc)
    if api_key.expires_at and api_key.expires_at < now:
        raise unauthorized("API key expired")

    # Atomic increment so concurrent requests don't lose counts
    await db.execute(
        update(ApiKey)
        .where(ApiKey.id == api_key.id)
        .values(request_count=ApiKey.request_count + 1, last_used_at=now)
        .execution_options(synchronize_session=False)
    )
    await db.commit()

    return api_key
