"""Authenticated, read-only canonical StudyMind entitlement status.

Does not activate purchases or imply that AI usage enforcement is enabled.
No Stripe secrets or billing identifiers are exposed to the client.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_auth
from app.db.engine import get_db
from app.config import get_settings
from app.db.models import StudyMindEntitlement, StudyMindUsageCounter, User
from app.usage.studymind_quota import FREE_LIMITS

router = APIRouter(prefix="/api", tags=["entitlements"])


@router.get("/me/entitlements")
async def my_entitlements(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    now = datetime.now(timezone.utc)
    rows = (await db.execute(
        select(StudyMindEntitlement).where(
            StudyMindEntitlement.user_id == current_user.id,
            StudyMindEntitlement.status == "active",
            StudyMindEntitlement.starts_at <= now,
            StudyMindEntitlement.expires_at > now,
        ).order_by(StudyMindEntitlement.expires_at.desc())
    )).scalars().all()
    if not rows:
        month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        free_counters = (await db.execute(select(StudyMindUsageCounter).where(
            StudyMindUsageCounter.user_id == current_user.id,
            StudyMindUsageCounter.period_start == month_start,
        ))).scalars().all()
        used = {c.feature: c.used for c in free_counters}
        features = {name: {"limit": limit, "used": used.get(name, 0),
                    "remaining": max(0, limit - used.get(name, 0))}
                    for name, limit in FREE_LIMITS.items()}
        return {"plan": "free", "expires_at": None, "features": features,
                "enforcement_enabled": get_settings().studymind_quota_enabled}

    # Multiple overlapping purchases are not summed until the billing policy is finalised.
    entitlement = rows[0]
    counters = (await db.execute(select(StudyMindUsageCounter).where(
        StudyMindUsageCounter.user_id == current_user.id,
        StudyMindUsageCounter.period_start >= entitlement.starts_at,
    ))).scalars().all()
    used = {}
    for counter in counters:
        used[counter.feature] = used.get(counter.feature, 0) + counter.used
    limits = entitlement.feature_limits or {}
    features = {
        name: {"limit": int(limit), "used": used.get(name, 0),
               "remaining": max(0, int(limit) - used.get(name, 0))}
        for name, limit in limits.items()
    }
    return {
        "plan": "paid", "expires_at": entitlement.expires_at.isoformat(),
        "features": features, "enforcement_enabled": False,
    }
