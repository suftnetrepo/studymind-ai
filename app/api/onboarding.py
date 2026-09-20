"""Role-based onboarding status."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_auth
from app.db.engine import get_db
from app.db.models import Enrolment, Module, User

router = APIRouter(prefix="/api", tags=["onboarding"])


@router.get("/onboarding/status")
async def get_onboarding_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    Student: done once they belong to an institution or are enrolled in a module.
    Lecturer / self-learner: done once they own a module.
    """
    if current_user.role == "student":
        enrolled = await db.scalar(
            select(func.count()).select_from(Enrolment)
            .where(Enrolment.student_id == current_user.id, Enrolment.status == "active")
        ) or 0
        return {
            "complete": current_user.institution_id is not None or enrolled > 0,
            "role": current_user.role,
            "module_count": enrolled,
        }

    if current_user.role == "admin":
        return {"complete": True, "role": current_user.role, "module_count": 0}

    owned = await db.scalar(
        select(func.count()).select_from(Module).where(Module.owner_id == current_user.id)
    ) or 0
    return {"complete": owned > 0, "role": current_user.role, "module_count": owned}
