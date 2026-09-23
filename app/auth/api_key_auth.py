"""
Authentication for v1 platform integrations (e.g. Learnify).

Two credentials, both sent as `Authorization: Bearer <token>` and stored only as SHA-256 hashes:
- API keys (`sm_live_...`) — server-to-server, never exposed to browsers.
- Session tokens (`st_...`) — minted from an API key via POST /api/v1/auth/session,
  short-lived and scoped to one platform course + user. Safe to hand to the browser.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_db
from app.db.models import ApiKey, SessionToken

KEY_PREFIX           = "sm_live_"
SESSION_TOKEN_PREFIX = "st_"
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


def generate_session_token() -> tuple[str, str]:
    """Returns (full_token, token_hash). Format: st_<64 hex chars>."""
    full_token = f"{SESSION_TOKEN_PREFIX}{secrets.token_hex(32)}"
    return full_token, hash_api_key(full_token)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _key_expired(api_key: ApiKey, now: datetime) -> bool:
    return api_key.expires_at is not None and api_key.expires_at < now


async def _record_usage(db: AsyncSession, api_key: ApiKey, now: datetime) -> None:
    # Atomic increment so concurrent requests don't lose counts
    await db.execute(
        update(ApiKey)
        .where(ApiKey.id == api_key.id)
        .values(request_count=ApiKey.request_count + 1, last_used_at=now)
        .execution_options(synchronize_session=False)
    )
    await db.commit()


async def get_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> ApiKey:
    """
    FastAPI dependency — validates the Bearer token as a platform API key.
    Use instead of require_auth for /api/v1 platform endpoints.
    """
    if not credentials or not credentials.credentials:
        raise _unauthorized("API key required")

    key = credentials.credentials
    if not key.startswith(KEY_PREFIX):
        raise _unauthorized("Invalid API key")

    result = await db.execute(
        select(ApiKey).where(
            ApiKey.key_hash == hash_api_key(key),
            ApiKey.is_active == True,  # noqa: E712
        )
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise _unauthorized("Invalid API key")

    now = datetime.now(timezone.utc)
    if _key_expired(api_key, now):
        raise _unauthorized("API key expired")

    await _record_usage(db, api_key, now)
    return api_key


@dataclass
class PlatformIdentity:
    """Who is calling a v1 endpoint. Session tokens pin course_id/user_id; API keys don't."""
    api_key:   ApiKey
    auth_type: Literal["api_key", "session_token"]
    course_id: Optional[str] = None
    user_id:   Optional[str] = None
    user_role: Optional[str] = None

    def scope(self, course_id: Optional[str], user_id: Optional[str]) -> tuple[str, str]:
        """
        Resolve the (course_id, user_id) a request acts on.
        Session tokens: always the token's scope; a conflicting body value is rejected.
        API keys: taken from the request, which must supply both.
        """
        if self.auth_type == "session_token":
            if (course_id and course_id != self.course_id) or (user_id and user_id != self.user_id):
                raise _unauthorized("Session token is not valid for this course or user")
            return self.course_id, self.user_id  # type: ignore[return-value]

        if not course_id or not user_id:
            raise HTTPException(status_code=422, detail="course_id and user_id are required")
        return course_id, user_id


async def get_platform_identity(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> PlatformIdentity:
    """
    FastAPI dependency for v1 platform endpoints — accepts an `sm_live_` API key
    (server-to-server) or an `st_` session token (browser).
    """
    if not credentials or not credentials.credentials:
        raise _unauthorized("Authentication required")

    token = credentials.credentials
    if not token.startswith(SESSION_TOKEN_PREFIX):
        api_key = await get_api_key(credentials, db)
        return PlatformIdentity(api_key=api_key, auth_type="api_key")

    result = await db.execute(
        select(SessionToken).where(SessionToken.token_hash == hash_api_key(token))
    )
    st = result.scalar_one_or_none()
    if not st:
        raise _unauthorized("Invalid session token")

    now = datetime.now(timezone.utc)
    if st.expires_at < now:
        raise _unauthorized("Session token expired")

    # Revoking or expiring the parent API key kills its session tokens immediately
    api_key = await db.get(ApiKey, st.api_key_id)
    if not api_key or not api_key.is_active or _key_expired(api_key, now):
        raise _unauthorized("Session token is no longer valid")

    await _record_usage(db, api_key, now)
    return PlatformIdentity(
        api_key=api_key,
        auth_type="session_token",
        course_id=st.course_id,
        user_id=st.user_id,
        user_role=st.user_role,
    )
