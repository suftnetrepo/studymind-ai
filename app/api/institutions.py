"""
Institution & User Management endpoints — Sprint 2.

INST-01: Super Admin creates institutions         POST /api/admin/institutions
INST-02: Admin manages departments                POST/GET /api/admin/institutions/{id}/departments
INST-03: Admin creates lecturer accounts          POST /api/admin/institutions/{id}/lecturers
INST-04: Bulk import students via CSV             POST /api/admin/institutions/{id}/students/bulk
INST-05: Student joins via institution code       POST /api/institutions/join
INST-06: Generate institution join code           POST /api/admin/institutions/{id}/codes
AUTH-08: Password reset request + confirm         POST /api/auth/password-reset/request|confirm
"""
from __future__ import annotations

import csv
import io
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_admin, require_auth
from app.auth.security import hash_password, hash_refresh_token, generate_refresh_token
from app.db.engine import get_db
from app.db.models import (
    BulkImportJob, Department, Institution,
    InstitutionCode, PasswordResetToken, User,
)
from app.db.schemas import (
    AdminUserListResponse,
    BulkImportResponse, BulkImportStatusResponse,
    CreateLecturerRequest,
    DepartmentCreate, DepartmentSchema,
    InstitutionCodeCreate, InstitutionCodeSchema,
    InstitutionCreate, InstitutionSchema,
    JoinWithCodeRequest,
    PasswordResetConfirm, PasswordResetRequestBody,
    UserSchema,
)
from app.logging_config import get_logger

log    = get_logger(__name__)
router = APIRouter(tags=["institutions"])


# ═══════════════════════════════════════════════════════════════════════════
# INST-01: Institutions (Super Admin)
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/api/admin/institutions", response_model=InstitutionSchema, status_code=201)
async def create_institution(
    body: InstitutionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """INST-01: Super Admin creates a new institution."""
    if body.domain:
        existing = await db.execute(
            select(Institution).where(Institution.domain == body.domain)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"An institution with domain '{body.domain}' already exists",
            )

    inst = Institution(
        id=uuid.uuid4(),
        name=body.name,
        domain=body.domain,
        tier=body.tier,
    )
    db.add(inst)
    await db.commit()
    await db.refresh(inst)
    log.info("institution_created", institution_id=str(inst.id), name=inst.name)
    return inst


@router.get("/api/admin/institutions", response_model=list[InstitutionSchema])
async def list_institutions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List all institutions. Super Admin only."""
    result = await db.execute(
        select(Institution).order_by(Institution.created_at.desc())
    )
    return result.scalars().all()


@router.get("/api/admin/institutions/{institution_id}", response_model=InstitutionSchema)
async def get_institution(
    institution_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    inst = await _get_institution_or_404(institution_id, db)
    return inst


@router.patch("/api/admin/institutions/{institution_id}", response_model=InstitutionSchema)
async def update_institution(
    institution_id: uuid.UUID,
    body: InstitutionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    inst = await _get_institution_or_404(institution_id, db)
    inst.name   = body.name
    inst.domain = body.domain
    inst.tier   = body.tier
    await db.commit()
    await db.refresh(inst)
    return inst


# ═══════════════════════════════════════════════════════════════════════════
# INST-02: Departments
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/api/admin/institutions/{institution_id}/departments",
    response_model=DepartmentSchema,
    status_code=201,
)
async def create_department(
    institution_id: uuid.UUID,
    body: DepartmentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """INST-02: Create a department within an institution."""
    await _get_institution_or_404(institution_id, db)

    dept = Department(
        id=uuid.uuid4(),
        institution_id=institution_id,
        name=body.name,
        code=body.code,
    )
    db.add(dept)
    await db.commit()
    await db.refresh(dept)
    log.info("department_created", dept_id=str(dept.id), name=dept.name)
    return dept


@router.get(
    "/api/admin/institutions/{institution_id}/departments",
    response_model=list[DepartmentSchema],
)
async def list_departments(
    institution_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """INST-02: List all departments in an institution."""
    await _get_institution_or_404(institution_id, db)
    result = await db.execute(
        select(Department)
        .where(Department.institution_id == institution_id)
        .order_by(Department.name)
    )
    return result.scalars().all()


@router.delete(
    "/api/admin/institutions/{institution_id}/departments/{dept_id}",
    status_code=204,
)
async def delete_department(
    institution_id: uuid.UUID,
    dept_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(
        select(Department).where(
            Department.id == dept_id,
            Department.institution_id == institution_id,
        )
    )
    dept = result.scalar_one_or_none()
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    await db.delete(dept)
    await db.commit()


# ═══════════════════════════════════════════════════════════════════════════
# INST-03: Create lecturer accounts
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/api/admin/institutions/{institution_id}/lecturers",
    response_model=UserSchema,
    status_code=201,
)
async def create_lecturer(
    institution_id: uuid.UUID,
    body: CreateLecturerRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """INST-03: Admin creates a lecturer account within an institution."""
    await _get_institution_or_404(institution_id, db)

    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )

    # Generate a temporary password — in production this would trigger an email invite
    temp_password = secrets.token_urlsafe(12) + "A1"

    lecturer = User(
        id=uuid.uuid4(),
        email=body.email,
        full_name=body.full_name,
        password_hash=hash_password(temp_password),
        role="lecturer",
        institution_id=institution_id,
        department_id=body.department_id,
        is_active=True,
        is_verified=False,
    )
    db.add(lecturer)
    await db.commit()
    await db.refresh(lecturer)

    log.info(
        "lecturer_created",
        lecturer_id=str(lecturer.id),
        institution_id=str(institution_id),
        temp_password_hint=f"{temp_password[:4]}****",  # log only for dev
    )
    # TODO Sprint 2 Phase 2: send email invite with temp_password
    return lecturer


@router.get(
    "/api/admin/institutions/{institution_id}/users",
    response_model=list[AdminUserListResponse],
)
async def list_institution_users(
    institution_id: uuid.UUID,
    role: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List all users in an institution, optionally filtered by role."""
    await _get_institution_or_404(institution_id, db)

    query = select(User).where(User.institution_id == institution_id)
    if role:
        query = query.where(User.role == role)
    query = query.order_by(User.created_at.desc())

    result = await db.execute(query)
    return result.scalars().all()


@router.patch("/api/admin/users/{user_id}/deactivate", status_code=200)
async def deactivate_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Admin deactivates a user account."""
    user = await _get_user_or_404(user_id, db)
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    user.is_active = False
    await db.commit()
    return {"message": f"User {user.email} deactivated"}


@router.patch("/api/admin/users/{user_id}/activate", status_code=200)
async def activate_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    user = await _get_user_or_404(user_id, db)
    user.is_active = True
    await db.commit()
    return {"message": f"User {user.email} activated"}


# ═══════════════════════════════════════════════════════════════════════════
# INST-04: Bulk import students via CSV
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/api/admin/institutions/{institution_id}/students/bulk",
    response_model=BulkImportResponse,
    status_code=202,
)
async def bulk_import_students(
    institution_id: uuid.UUID,
    file: UploadFile = File(..., description="CSV file: email, full_name, [student_id], [department_code]"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    INST-04: Bulk import students from a CSV file.

    Expected CSV columns: email, full_name (required) | student_id, department_code (optional)
    Max 1000 rows per import.
    """
    await _get_institution_or_404(institution_id, db)

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are accepted")

    content = await file.read()
    try:
        text   = content.decode("utf-8-sig")  # handle BOM
        reader = csv.DictReader(io.StringIO(text))
        rows   = list(reader)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    if not rows:
        raise HTTPException(status_code=422, detail="CSV file is empty")
    if len(rows) > 1000:
        raise HTTPException(status_code=422, detail="Maximum 1000 rows per import")

    required_cols = {"email", "full_name"}
    if not required_cols.issubset(set(rows[0].keys())):
        raise HTTPException(
            status_code=422,
            detail=f"CSV must contain columns: {required_cols}. Found: {set(rows[0].keys())}",
        )

    # Create the job record
    job = BulkImportJob(
        id=uuid.uuid4(),
        institution_id=institution_id,
        created_by=current_user.id,
        status="processing",
        total_rows=len(rows),
    )
    db.add(job)
    await db.flush()

    # Process rows synchronously for MVP (background task in production)
    success_count = 0
    errors        = []

    for i, row in enumerate(rows, start=1):
        email     = (row.get("email") or "").strip().lower()
        full_name = (row.get("full_name") or "").strip()

        if not email or not full_name:
            errors.append({"row": i, "error": "email and full_name are required", "data": row})
            continue

        # Check existing
        existing = await db.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none():
            errors.append({"row": i, "error": f"Email already registered: {email}", "data": row})
            continue

        temp_password = secrets.token_urlsafe(10) + "A1"
        student = User(
            id=uuid.uuid4(),
            email=email,
            full_name=full_name,
            password_hash=hash_password(temp_password),
            role="student",
            institution_id=institution_id,
            is_active=True,
            is_verified=False,
            profile={"student_id": row.get("student_id", "").strip() or None},
        )
        db.add(student)
        success_count += 1

    job.status        = "complete"
    job.success_count = success_count
    job.error_count   = len(errors)
    job.errors        = errors or None
    job.completed_at  = datetime.now(timezone.utc)

    await db.commit()

    log.info(
        "bulk_import_complete",
        job_id=str(job.id),
        total=len(rows),
        success=success_count,
        errors=len(errors),
    )

    return BulkImportResponse(
        job_id=job.id,
        status=job.status,
        total_rows=len(rows),
        message=f"Import complete: {success_count} created, {len(errors)} errors",
    )


@router.get(
    "/api/admin/bulk-imports/{job_id}",
    response_model=BulkImportStatusResponse,
)
async def get_bulk_import_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(select(BulkImportJob).where(BulkImportJob.id == job_id))
    job    = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Import job not found")
    return job


# ═══════════════════════════════════════════════════════════════════════════
# INST-05 + INST-06: Institution join codes
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/api/admin/institutions/{institution_id}/codes",
    response_model=InstitutionCodeSchema,
    status_code=201,
)
async def create_institution_code(
    institution_id: uuid.UUID,
    body: InstitutionCodeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """INST-05 / INST-06: Generate a shareable join or enrolment code."""
    await _get_institution_or_404(institution_id, db)

    # Generate a short, readable code
    raw_code = secrets.token_urlsafe(6).upper()[:8]

    expires_at = None
    if body.expires_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_days)

    code = InstitutionCode(
        id=uuid.uuid4(),
        institution_id=institution_id,
        code=raw_code,
        code_type=body.code_type,
        target_role=body.target_role,
        max_uses=body.max_uses,
        expires_at=expires_at,
        created_by=current_user.id,
        is_active=True,
    )
    db.add(code)
    await db.commit()
    await db.refresh(code)

    log.info("institution_code_created", code=raw_code, type=body.code_type)
    return code


@router.get(
    "/api/admin/institutions/{institution_id}/codes",
    response_model=list[InstitutionCodeSchema],
)
async def list_institution_codes(
    institution_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(
        select(InstitutionCode)
        .where(InstitutionCode.institution_id == institution_id)
        .order_by(InstitutionCode.created_at.desc())
    )
    return result.scalars().all()


@router.post("/api/institutions/join", response_model=UserSchema)
async def join_with_code(
    body: JoinWithCodeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    INST-05: Authenticated user joins an institution using a code.
    Typically called right after registration.
    """
    result = await db.execute(
        select(InstitutionCode).where(
            InstitutionCode.code == body.code.upper(),
            InstitutionCode.is_active == True,
            InstitutionCode.code_type == "institution_join",
        )
    )
    code = result.scalar_one_or_none()

    if not code:
        raise HTTPException(status_code=404, detail="Invalid or inactive code")

    # Check expiry
    if code.expires_at and code.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="This code has expired")

    # Check usage limit
    if code.max_uses and code.use_count >= code.max_uses:
        raise HTTPException(status_code=410, detail="This code has reached its maximum uses")

    # Check user not already in an institution
    if current_user.institution_id:
        raise HTTPException(
            status_code=409,
            detail="You are already a member of an institution",
        )

    # Update user
    current_user.institution_id = code.institution_id
    if code.target_role and current_user.role == "self_learner":
        current_user.role = code.target_role

    # Increment use count
    code.use_count += 1
    if code.max_uses and code.use_count >= code.max_uses:
        code.is_active = False

    await db.commit()
    await db.refresh(current_user)

    log.info(
        "user_joined_institution",
        user_id=str(current_user.id),
        institution_id=str(code.institution_id),
    )
    return current_user


# ═══════════════════════════════════════════════════════════════════════════
# AUTH-08: Password reset
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/api/auth/password-reset/request", status_code=202)
async def request_password_reset(
    body: PasswordResetRequestBody,
    db: AsyncSession = Depends(get_db),
):
    """
    AUTH-08: Request a password reset.
    Always returns 202 — does not reveal whether the email exists (security best practice).
    In production this sends an email; in MVP the token is returned in the response for testing.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user   = result.scalar_one_or_none()

    if not user:
        # Return 202 regardless — do not reveal user existence
        return {"message": "If that email exists, a reset link has been sent"}

    # Generate reset token
    raw_token   = secrets.token_urlsafe(32)
    token_hash  = hash_refresh_token(raw_token)
    expires_at  = datetime.now(timezone.utc) + timedelta(hours=1)

    # Invalidate any existing tokens for this user
    existing_tokens = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used == False,
        )
    )
    for t in existing_tokens.scalars().all():
        t.used = True

    reset_token = PasswordResetToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    db.add(reset_token)
    await db.commit()

    log.info("password_reset_requested", user_id=str(user.id))

    # TODO production: send email with reset link containing raw_token
    # For MVP/dev: return token in response body
    return {
        "message": "If that email exists, a reset link has been sent",
        "dev_token": raw_token,  # REMOVE IN PRODUCTION
    }


@router.post("/api/auth/password-reset/confirm", status_code=200)
async def confirm_password_reset(
    body: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db),
):
    """AUTH-08: Submit the reset token and new password."""
    token_hash = hash_refresh_token(body.token)

    result = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.used == False,
        )
    )
    reset_token = result.scalar_one_or_none()

    invalid_exc = HTTPException(
        status_code=400, detail="Invalid or expired reset token"
    )

    if not reset_token:
        raise invalid_exc

    if reset_token.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        reset_token.used = True
        await db.commit()
        raise invalid_exc

    # Update password
    user_result = await db.execute(select(User).where(User.id == reset_token.user_id))
    user        = user_result.scalar_one_or_none()
    if not user:
        raise invalid_exc

    user.password_hash = hash_password(body.new_password)
    reset_token.used   = True

    # Revoke all refresh tokens (force re-login)
    tokens = await db.execute(
        select(type("RefreshToken", (), {"user_id": None}))
    )
    await db.commit()

    log.info("password_reset_complete", user_id=str(user.id))
    return {"message": "Password reset successfully. Please log in with your new password."}


# ═══════════════════════════════════════════════════════════════════════════
# Admin stats (ANLX-05 preview)
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/admin/stats")
async def admin_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """High-level platform statistics for Super Admin."""
    total_institutions = (await db.execute(select(func.count()).select_from(Institution))).scalar() or 0
    total_users        = (await db.execute(select(func.count()).select_from(User))).scalar() or 0
    total_students     = (await db.execute(
        select(func.count()).select_from(User).where(User.role == "student")
    )).scalar() or 0
    total_lecturers    = (await db.execute(
        select(func.count()).select_from(User).where(User.role == "lecturer")
    )).scalar() or 0

    return {
        "total_institutions": total_institutions,
        "total_users":        total_users,
        "total_students":     total_students,
        "total_lecturers":    total_lecturers,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

async def _get_institution_or_404(institution_id: uuid.UUID, db: AsyncSession) -> Institution:
    result = await db.execute(select(Institution).where(Institution.id == institution_id))
    inst   = result.scalar_one_or_none()
    if not inst:
        raise HTTPException(status_code=404, detail="Institution not found")
    return inst


async def _get_user_or_404(user_id: uuid.UUID, db: AsyncSession) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user   = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
