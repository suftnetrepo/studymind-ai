"""Study streak and activity summary."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_auth
from app.db.engine import get_db
from app.db.models import (
    ChatSession, FlashcardDeck, Module, QuizAttempt, StudyActivity, Summary, User,
)

router = APIRouter(prefix="/api", tags=["activity"])


@router.get("/activity/streak")
async def get_streak(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    today      = datetime.now(timezone.utc).date()
    week_start = today - timedelta(days=today.weekday())   # Monday
    week_end   = week_start + timedelta(days=6)

    result = await db.execute(
        select(distinct(StudyActivity.activity_date))
        .where(
            StudyActivity.user_id == current_user.id,
            StudyActivity.activity_date >= today - timedelta(days=366),
        )
    )
    active_dates = sorted((row[0] for row in result.all()), reverse=True)

    # Consecutive days ending today, or yesterday if the user hasn't studied yet today.
    streak, expected = 0, today
    if active_dates and active_dates[0] == today - timedelta(days=1):
        expected = today - timedelta(days=1)
    for d in active_dates:
        if d == expected:
            streak  += 1
            expected = d - timedelta(days=1)
        else:
            break

    week_active = [d.isoformat() for d in active_dates if week_start <= d <= week_end]

    breakdown_rows = await db.execute(
        select(StudyActivity.activity_type, func.count(distinct(StudyActivity.activity_date)))
        .where(StudyActivity.user_id == current_user.id, StudyActivity.activity_date >= week_start)
        .group_by(StudyActivity.activity_type)
    )
    total = await db.scalar(
        select(func.count()).select_from(StudyActivity).where(StudyActivity.user_id == current_user.id)
    )

    recent = (await db.execute(
        select(StudyActivity)
        .where(StudyActivity.user_id == current_user.id)
        .order_by(StudyActivity.created_at.desc())
        .limit(20)
    )).scalars().all()

    module_ids = {a.module_id for a in recent if a.module_id}
    module_map: dict[str, dict] = {}
    if module_ids:
        mods = await db.execute(
            select(Module.id, Module.title, Module.course_code).where(Module.id.in_(module_ids))
        )
        module_map = {str(r[0]): {"title": r[1], "course_code": r[2]} for r in mods.all()}

    recent_activities = []
    for a in recent:
        mod = module_map.get(str(a.module_id)) if a.module_id else None
        recent_activities.append({
            "id":           str(a.id),
            "type":         a.activity_type,
            "module_id":    str(a.module_id) if a.module_id else None,
            "course_code":  mod["course_code"] if mod else None,
            "module_title": mod["title"] if mod else None,
            "date":         a.activity_date.isoformat(),
            "created_at":   a.created_at.isoformat(),
        })

    return {
        "recent_activities": recent_activities,
        "streak_days":      streak,
        "active_dates":     [d.isoformat() for d in active_dates[:30]],
        "week_active":      week_active,
        "week_start":       week_start.isoformat(),
        "weekly_count":     len(week_active),
        "weekly_target":    7,
        "weekly_progress":  round(len(week_active) / 7 * 100),
        "breakdown":        {k: v for k, v in breakdown_rows.all()},
        "total_activities": total or 0,
    }


@router.get("/activity/summary")
async def get_activity_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """Real totals from the study tables (activity rows are one-per-day, so they undercount)."""
    async def count(model) -> int:
        return await db.scalar(
            select(func.count()).select_from(model).where(model.user_id == current_user.id)
        ) or 0

    rows = await db.execute(
        select(StudyActivity.activity_type, func.count())
        .where(StudyActivity.user_id == current_user.id)
        .group_by(StudyActivity.activity_type)
    )
    logged = {k: v for k, v in rows.all()}

    out = {
        "chat":            await count(ChatSession),
        "quiz":            await count(QuizAttempt),
        "flashcard":       await count(FlashcardDeck),
        "summary":         await count(Summary),
        "document_upload": logged.get("document_upload", 0),
        "notes":           logged.get("notes", 0),
    }
    out["total"] = sum(out.values())
    return out


@router.get("/quota/status")
async def get_quota_status(current_user: User = Depends(require_auth)):
    # Quotas are enforced on the device via RevenueCat; the backend only reports the account context.
    is_institution_user = current_user.role in ("student", "lecturer", "admin")
    return {
        "is_pro":              is_institution_user,
        "is_institution_user": is_institution_user,
        "role":                current_user.role,
        "quotas":              {},
    }
