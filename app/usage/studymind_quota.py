"""StudyMind canonical-user quota primitives (NOT wired to AI routes yet).

This module deliberately has no free-tier defaults. Product-approved limits and
endpoint integration are required before switching on enforcement.
Reservations use one PostgreSQL transaction and a per-user advisory transaction lock
so simultaneous requests cannot overspend a user's feature allowance.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import text

from app.db.models import StudyMindEntitlement, StudyMindUsageCounter


class StudyMindQuotaExceeded(Exception):
    pass


def _lock_id(user_id: UUID) -> int:
    # Stable signed 64-bit lock key. This lock is held until transaction commit/rollback.
    raw = hashlib.sha256(user_id.bytes).digest()[:8]
    return int.from_bytes(raw, 'big', signed=True)


async def reserve_paid_feature(db: AsyncSession, user_id: UUID, feature: str) -> bool:
    """Atomically reserve one paid feature use. Caller MUST commit before invoking AI.

    Returns False when no active paid entitlement exists (free-tier rules belong
    to a separate, explicit policy). Raises when the paid allowance is exhausted.
    This helper must be used with a dedicated short-lived transaction, never a
    transaction that remains open during model inference.
    """
    if feature not in ('chat', 'quiz', 'flashcards', 'summary'):
        raise ValueError('Unsupported AI feature')
    await db.execute(text('SELECT pg_advisory_xact_lock(:key)'), {'key': _lock_id(user_id)})
    now = datetime.now(timezone.utc)
    entitlement = (await db.execute(
        select(StudyMindEntitlement).where(
            StudyMindEntitlement.user_id == user_id,
            StudyMindEntitlement.status == 'active',
            StudyMindEntitlement.starts_at <= now,
            StudyMindEntitlement.expires_at > now,
        ).order_by(StudyMindEntitlement.expires_at.desc()).limit(1)
    )).scalar_one_or_none()
    if entitlement is None:
        return False
    limit = (entitlement.feature_limits or {}).get(feature)
    if type(limit) is not int or limit < 0:
        raise StudyMindQuotaExceeded('Feature is not included in this pass')
    period = entitlement.starts_at
    await db.execute(pg_insert(StudyMindUsageCounter).values(
        user_id=user_id, feature=feature, period_start=period, used=0
    ).on_conflict_do_nothing())
    counter = (await db.execute(select(StudyMindUsageCounter).where(
        StudyMindUsageCounter.user_id == user_id,
        StudyMindUsageCounter.feature == feature,
        StudyMindUsageCounter.period_start == period,
    ).with_for_update())).scalar_one()
    if counter.used >= limit:
        raise StudyMindQuotaExceeded('AI usage allowance exhausted')
    counter.used += 1
    await db.flush()
    return True
