"""Opt-in canonical StudyMind AI quota guard.

Reservations commit in their own short transaction BEFORE AI work. Failed
operations are compensated in another transaction. The rollout flag is OFF by
default; a durable reservation ledger is required for crash-safe exactly-once
accounting before production enablement.
"""
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import HTTPException
from app.config import get_settings
from app.db.engine import AsyncSessionLocal
from app.usage.studymind_quota import (
    StudyMindQuotaExceeded, reserve_feature, refund_feature,
)


@asynccontextmanager
async def metered_generation(user_id: UUID, feature: str):
    if not get_settings().studymind_quota_enabled:
        yield
        return
    reservation = None
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                reservation = await reserve_feature(session, user_id, feature)
    except StudyMindQuotaExceeded:
        raise HTTPException(status_code=429, detail="AI allowance exhausted. Upgrade on StudyMind Web to continue.")
    try:
        yield
    except BaseException:
        if reservation:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await refund_feature(session, user_id, feature, reservation["period_start"])
        raise
