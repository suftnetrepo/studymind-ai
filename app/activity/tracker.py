"""Study-activity logging for streaks."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.engine import AsyncSessionLocal
from app.db.models import StudyActivity
from app.logging_config import get_logger

log = get_logger(__name__)


async def log_activity(
    user_id: uuid.UUID,
    activity_type: str,
    module_id: uuid.UUID | None = None,
) -> None:
    """
    Record that the user studied today. Idempotent per user/type/module/day.
    Uses its own session so a failure here can never affect the caller's request.
    """
    today = datetime.now(timezone.utc).date()
    try:
        async with AsyncSessionLocal() as session:
            exists = await session.execute(
                select(StudyActivity.id).where(
                    StudyActivity.user_id == user_id,
                    StudyActivity.activity_type == activity_type,
                    StudyActivity.activity_date == today,
                    StudyActivity.module_id.is_(None) if module_id is None
                    else StudyActivity.module_id == module_id,
                ).limit(1)
            )
            if exists.first():
                return
            await session.execute(
                pg_insert(StudyActivity).values(
                    id=uuid.uuid4(), user_id=user_id, activity_type=activity_type,
                    module_id=module_id, activity_date=today,
                ).on_conflict_do_nothing()
            )
            await session.commit()
    except Exception as exc:
        log.warning("log_activity_failed", error=str(exc), activity_type=activity_type)
