"""
v1 API key management — admin only (JWT auth).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.api_key_auth import generate_api_key
from app.auth.dependencies import require_admin
from app.db.engine import get_db
from app.db.models import ApiKey, PlatformCourse, User
from app.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/v1/admin", tags=["API Key Admin"])


class CreateApiKeyRequest(BaseModel):
    name:        str = Field(..., min_length=1, max_length=255)
    platform:    str = Field(..., min_length=2, max_length=64, pattern="^[a-z0-9_-]+$")
    owner_email: str = Field(..., min_length=3, max_length=320)
    expires_at:  Optional[datetime] = None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


async def _get_key_or_404(key_id: uuid.UUID, db: AsyncSession) -> ApiKey:
    key = await db.get(ApiKey, key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    return key


@router.post("/api-keys", status_code=201)
async def create_api_key(
    req:          CreateApiKeyRequest,
    db:           AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Create a new API key for a platform. The full key is returned ONCE."""
    full_key, key_hash, key_prefix = generate_api_key()

    api_key = ApiKey(
        id=uuid.uuid4(),
        name=req.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        platform=req.platform,
        owner_email=req.owner_email,
        is_active=True,
        expires_at=req.expires_at,
    )
    db.add(api_key)
    await db.commit()

    log.info("api_key_created", key_id=str(api_key.id), platform=req.platform, by=str(current_user.id))

    return {
        "id":         str(api_key.id),
        "key":        full_key,
        "prefix":     key_prefix,
        "platform":   req.platform,
        "expires_at": _iso(req.expires_at),
        "message":    "Store this key securely — it will not be shown again.",
    }


@router.get("/api-keys")
async def list_api_keys(
    db:           AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List all API keys (never includes the key itself)."""
    result = await db.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
    return [{
        "id":            str(k.id),
        "name":          k.name,
        "prefix":        k.key_prefix,
        "platform":      k.platform,
        "owner_email":   k.owner_email,
        "is_active":     k.is_active,
        "request_count": k.request_count,
        "last_used_at":  _iso(k.last_used_at),
        "created_at":    _iso(k.created_at),
        "expires_at":    _iso(k.expires_at),
    } for k in result.scalars().all()]


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id:       uuid.UUID,
    db:           AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Revoke an API key (soft — keeps usage history)."""
    key = await _get_key_or_404(key_id, db)
    key.is_active = False
    await db.commit()
    log.info("api_key_revoked", key_id=str(key_id), by=str(current_user.id))
    return {"revoked": True}


@router.get("/api-keys/{key_id}/usage")
async def get_api_key_usage(
    key_id:       uuid.UUID,
    db:           AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Usage stats and ingested courses for an API key."""
    key = await _get_key_or_404(key_id, db)

    courses_result = await db.execute(
        select(PlatformCourse)
        .where(PlatformCourse.api_key_id == key.id)
        .order_by(PlatformCourse.created_at.desc())
    )

    return {
        "key_id":        str(key.id),
        "platform":      key.platform,
        "is_active":     key.is_active,
        "request_count": key.request_count,
        "last_used_at":  _iso(key.last_used_at),
        "courses": [{
            "course_id":   c.platform_course_id,
            "user_id":     c.platform_user_id,
            "title":       c.course_title,
            "status":      c.status,
            "chunk_count": c.chunk_count,
            "indexed_at":  _iso(c.indexed_at),
        } for c in courses_result.scalars().all()],
    }
