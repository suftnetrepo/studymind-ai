"""
Assignment endpoints.

Lecturer:
  POST   /api/assignments                              — create (brief + marking scheme upload)
  GET    /api/assignments?module_id=<id>               — list assignments for a module
  GET    /api/assignments/{id}                         — assignment detail + brief download URL
  GET    /api/assignments/{id}/submissions             — list all student submissions
  GET    /api/assignments/{id}/submissions/{sub_id}    — single submission detail + download URL

Student:
  GET    /api/assignments/student?module_id=<id>       — list module assignments (no scheme)
  GET    /api/assignments/{id}/brief                   — get brief download URL
  POST   /api/assignments/{id}/submit                  — upload completed work
  GET    /api/assignments/{id}/my-submission           — own submission + grading result
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.grading import grade_submission
from app.auth.dependencies import require_auth, require_lecturer, require_student
from app.db.engine import get_db
from app.db.models import Assignment, AssignmentSubmission, Enrolment, Module, User
from app.db.schemas import (
    AssignmentDetailSchema,
    AssignmentSchema,
    SubmissionDetailSchema,
    SubmissionSchema,
)
from app.logging_config import get_logger
from app.storage import cloudinary_client as storage

log    = get_logger(__name__)
router = APIRouter(prefix="/api/assignments", tags=["assignments"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc"}
MAX_FILE_SIZE      = 50 * 1024 * 1024  # 50 MB

PLATFORM = "studymind"


async def _store(file_bytes: bytes, filename: str, course_id: str, document_id: str) -> dict:
    """Upload to file storage; turn storage failures into clear HTTP errors instead of a bare 500."""
    try:
        return await storage.upload_document(
            file_bytes, filename, platform=PLATFORM, course_id=course_id, document_id=document_id,
        )
    except storage.StorageNotConfigured:
        log.error("assignment_storage_not_configured", document_id=document_id)
        raise HTTPException(status_code=503, detail="File storage is not available right now. Please contact support.")
    except Exception:
        log.exception("assignment_upload_failed", document_id=document_id)
        raise HTTPException(status_code=502, detail="The file could not be uploaded. Please try again.")


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _require_allowed(filename: str) -> None:
    if _ext(filename) not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only PDF and Word files are accepted. Got: .{_ext(filename)}",
        )


async def _read_upload(file: UploadFile) -> bytes:
    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File must be under 50 MB.",
        )
    return raw


# ── Background grading ──────────────────────────────────────────────────────

async def _grade_in_background(
    submission_id: uuid.UUID,
    brief_url: str,
    brief_filename: str,
    scheme_url: str,
    scheme_filename: str,
    submission_url: str,
    submission_filename: str,
    grading_prompt: Optional[str] = None,
) -> None:
    """Run AI grading and persist the result — called as a background task."""
    from app.db.engine import AsyncSessionLocal  # local import to avoid circular
    async with AsyncSessionLocal() as db:
        sub = await db.get(AssignmentSubmission, submission_id)
        if not sub:
            log.error("grading_submission_missing", submission_id=str(submission_id))
            return

        sub.status = "grading"
        await db.commit()

        # Re-fetch assignment to get public_ids for signed URL generation
        assignment = await db.get(Assignment, sub.assignment_id)
        if not assignment:
            log.error("grading_assignment_missing", submission_id=str(submission_id))
            return

        brief_dl  = storage.download_url(assignment.brief_cloudinary_id,  assignment.brief_cloudinary_url)
        scheme_dl = storage.download_url(assignment.scheme_cloudinary_id, assignment.scheme_cloudinary_url)
        sub_dl    = storage.download_url(sub.cloudinary_id, sub.cloudinary_url)

        result = await grade_submission(
            brief_url=brief_dl,
            brief_filename=brief_filename,
            scheme_url=scheme_dl,
            scheme_filename=scheme_filename,
            submission_url=sub_dl,
            submission_filename=submission_filename,
            grading_prompt=assignment.grading_prompt,
        )

        sub.score           = result.score
        sub.strengths       = result.strengths
        sub.weaknesses      = result.weaknesses
        sub.recommendations = result.recommendations
        sub.grading_notes   = result.notes
        sub.status          = "graded" if not result.notes.startswith(("Download failed", "Grading AI error")) else "error"
        sub.graded_at       = datetime.now(timezone.utc)
        await db.commit()
        log.info("grading_complete", submission_id=str(submission_id), score=result.score, status=sub.status)


# ── Lecturer: create assignment ─────────────────────────────────────────────

@router.post("", response_model=AssignmentSchema, status_code=201)
async def create_assignment(
    module_id:      uuid.UUID = Form(...),
    title:          str        = Form(...),
    description:    str        = Form(""),
    due_at:         datetime   = Form(...),
    max_score:      int        = Form(100),
    grading_prompt: str        = Form(""),
    brief:          UploadFile = File(...),
    scheme:         UploadFile = File(...),
    db:             AsyncSession = Depends(get_db),
    current_user:   User         = Depends(require_lecturer),
):
    """Create an assignment with brief + marking scheme PDFs/Word files."""
    _require_allowed(brief.filename or "")
    _require_allowed(scheme.filename or "")

    # Confirm lecturer owns this module
    module = await db.get(Module, module_id)
    if not module or module.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Module not found or not yours.")

    brief_bytes  = await _read_upload(brief)
    scheme_bytes = await _read_upload(scheme)

    assignment_id = uuid.uuid4()

    brief_result = await _store(brief_bytes, brief.filename or "brief", str(module_id), f"{assignment_id}_brief")
    try:
        scheme_result = await _store(scheme_bytes, scheme.filename or "scheme", str(module_id), f"{assignment_id}_scheme")
    except HTTPException:
        # Don't leave the already-uploaded brief orphaned in storage (delete_document never raises).
        await storage.delete_document(brief_result["public_id"], brief_result["secure_url"])
        raise

    assignment = Assignment(
        id=assignment_id,
        module_id=module_id,
        created_by=current_user.id,
        title=title.strip(),
        description=description.strip() or None,
        due_at=due_at,
        max_score=max(1, min(100, max_score)),
        grading_prompt=grading_prompt.strip() or None,
        brief_cloudinary_id=brief_result["public_id"],
        brief_cloudinary_url=brief_result["secure_url"],
        brief_filename=brief.filename or "brief",
        brief_file_size=brief_result["bytes"],
        scheme_cloudinary_id=scheme_result["public_id"],
        scheme_cloudinary_url=scheme_result["secure_url"],
        scheme_filename=scheme.filename or "scheme",
        scheme_file_size=scheme_result["bytes"],
    )
    db.add(assignment)
    await db.commit()
    await db.refresh(assignment)

    schema = AssignmentSchema.model_validate(assignment)
    schema.submission_count = 0
    return schema


# ── Lecturer: list assignments for module ──────────────────────────────────

@router.get("/lecturer", response_model=list[AssignmentSchema])
async def list_assignments_lecturer(
    module_id:    uuid.UUID = Query(...),
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(require_lecturer),
):
    module = await db.get(Module, module_id)
    if not module or module.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Module not found or not yours.")

    result = await db.execute(
        select(Assignment).where(Assignment.module_id == module_id)
        .order_by(Assignment.due_at)
    )
    assignments = result.scalars().all()

    schemas = []
    for a in assignments:
        sub_count = await db.scalar(
            select(func.count()).where(AssignmentSubmission.assignment_id == a.id)
        )
        s = AssignmentSchema.model_validate(a)
        s.submission_count = sub_count or 0
        schemas.append(s)
    return schemas


# ── Lecturer: list all submissions for an assignment ───────────────────────

@router.get("/{assignment_id}/submissions", response_model=list[SubmissionDetailSchema])
async def list_submissions(
    assignment_id: uuid.UUID,
    db:            AsyncSession = Depends(get_db),
    current_user:  User          = Depends(require_lecturer),
):
    assignment = await db.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    module = await db.get(Module, assignment.module_id)
    if not module or module.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your module.")

    result = await db.execute(
        select(AssignmentSubmission)
        .where(AssignmentSubmission.assignment_id == assignment_id)
        .order_by(AssignmentSubmission.submitted_at)
    )
    subs = result.scalars().all()

    schemas = []
    for sub in subs:
        student = await db.get(User, sub.student_id)
        dl_url = storage.download_url(sub.cloudinary_id, sub.cloudinary_url)
        s = SubmissionDetailSchema.model_validate(sub)
        s.student_name  = student.full_name if student else ""
        s.student_email = student.email      if student else ""
        s.file_download_url = dl_url
        schemas.append(s)
    return schemas


# ── Lecturer: single submission detail ────────────────────────────────────

@router.get("/{assignment_id}/submissions/{submission_id}", response_model=SubmissionDetailSchema)
async def get_submission(
    assignment_id:  uuid.UUID,
    submission_id:  uuid.UUID,
    db:             AsyncSession = Depends(get_db),
    current_user:   User          = Depends(require_lecturer),
):
    assignment = await db.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    module = await db.get(Module, assignment.module_id)
    if not module or module.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your module.")

    sub = await db.get(AssignmentSubmission, submission_id)
    if not sub or sub.assignment_id != assignment_id:
        raise HTTPException(status_code=404, detail="Submission not found.")

    student = await db.get(User, sub.student_id)
    dl_url = storage.download_url(sub.cloudinary_id, sub.cloudinary_url)
    s = SubmissionDetailSchema.model_validate(sub)
    s.student_name  = student.full_name if student else ""
    s.student_email = student.email      if student else ""
    s.file_download_url = dl_url
    return s


# ── Lecturer: get assignment detail (includes marking scheme download URL) ─

@router.get("/lecturer/{assignment_id}", response_model=AssignmentDetailSchema)
async def get_assignment_lecturer(
    assignment_id: uuid.UUID,
    db:            AsyncSession = Depends(get_db),
    current_user:  User          = Depends(require_lecturer),
):
    assignment = await db.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    module = await db.get(Module, assignment.module_id)
    if not module or module.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your module.")

    sub_count = await db.scalar(
        select(func.count()).where(AssignmentSubmission.assignment_id == assignment_id)
    )
    brief_url = storage.download_url(assignment.brief_cloudinary_id, assignment.brief_cloudinary_url)
    s = AssignmentDetailSchema.model_validate(assignment)
    s.submission_count  = sub_count or 0
    s.brief_download_url = brief_url
    return s


# ── Student: list assignments for enrolled module ─────────────────────────

@router.get("/student", response_model=list[AssignmentSchema])
async def list_assignments_student(
    module_id:    uuid.UUID = Query(...),
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(require_student),
):
    # Verify student is enrolled
    enrolment = await db.scalar(
        select(Enrolment).where(
            Enrolment.student_id == current_user.id,
            Enrolment.module_id  == module_id,
        )
    )
    if not enrolment:
        raise HTTPException(status_code=403, detail="Not enrolled in this module.")

    result = await db.execute(
        select(Assignment)
        .where(Assignment.module_id == module_id, Assignment.status == "active")
        .order_by(Assignment.due_at)
    )
    assignments = result.scalars().all()

    schemas = []
    for a in assignments:
        # Check if student has already submitted
        my_sub = await db.scalar(
            select(AssignmentSubmission).where(
                AssignmentSubmission.assignment_id == a.id,
                AssignmentSubmission.student_id    == current_user.id,
            )
        )
        s = AssignmentSchema.model_validate(a)
        if my_sub:
            s.my_submission_id     = my_sub.id
            s.my_submission_status = my_sub.status
        schemas.append(s)
    return schemas


# ── Student: get brief download URL ────────────────────────────────────────

@router.get("/{assignment_id}/brief")
async def get_brief_url(
    assignment_id: uuid.UUID,
    db:            AsyncSession = Depends(get_db),
    current_user:  User          = Depends(require_auth),
):
    assignment = await db.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found.")

    # Lecturers who own the module can access; students who are enrolled can access
    if current_user.role in ("lecturer", "admin"):
        module = await db.get(Module, assignment.module_id)
        if not module or module.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not your module.")
    else:
        enrolment = await db.scalar(
            select(Enrolment).where(
                Enrolment.student_id == current_user.id,
                Enrolment.module_id  == assignment.module_id,
            )
        )
        if not enrolment:
            raise HTTPException(status_code=403, detail="Not enrolled in this module.")

    url = storage.download_url(assignment.brief_cloudinary_id, assignment.brief_cloudinary_url)
    return {"download_url": url, "filename": assignment.brief_filename}


# ── Student: submit assignment ─────────────────────────────────────────────

@router.post("/{assignment_id}/submit", response_model=SubmissionSchema, status_code=201)
async def submit_assignment(
    assignment_id:   uuid.UUID,
    background_tasks: BackgroundTasks,
    work:            UploadFile = File(...),
    db:              AsyncSession = Depends(get_db),
    current_user:    User          = Depends(require_student),
):
    _require_allowed(work.filename or "")

    assignment = await db.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found.")

    # Must be enrolled
    enrolment = await db.scalar(
        select(Enrolment).where(
            Enrolment.student_id == current_user.id,
            Enrolment.module_id  == assignment.module_id,
        )
    )
    if not enrolment:
        raise HTTPException(status_code=403, detail="Not enrolled in this module.")

    # Check deadline — block late submissions
    now = datetime.now(timezone.utc)
    due_at = assignment.due_at if assignment.due_at.tzinfo is not None else assignment.due_at.replace(tzinfo=timezone.utc)
    if now > due_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The deadline has passed. Late submissions are not accepted.",
        )

    # Prevent duplicate submissions
    existing = await db.scalar(
        select(AssignmentSubmission).where(
            AssignmentSubmission.assignment_id == assignment_id,
            AssignmentSubmission.student_id    == current_user.id,
        )
    )
    if existing:
        raise HTTPException(status_code=409, detail="You have already submitted this assignment.")

    work_bytes = await _read_upload(work)
    sub_id = uuid.uuid4()

    upload_result = await _store(work_bytes, work.filename or "submission", str(assignment.module_id), f"{assignment_id}_sub_{sub_id}")

    submission = AssignmentSubmission(
        id=sub_id,
        assignment_id=assignment_id,
        student_id=current_user.id,
        cloudinary_id=upload_result["public_id"],
        cloudinary_url=upload_result["secure_url"],
        filename=work.filename or "submission",
        file_size=upload_result["bytes"],
        is_late=False,
        status="pending",
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    # Kick off grading in the background
    background_tasks.add_task(
        _grade_in_background,
        sub_id,
        assignment.brief_cloudinary_url,
        assignment.brief_filename,
        assignment.scheme_cloudinary_url,
        assignment.scheme_filename,
        upload_result["secure_url"],
        work.filename or "submission",
        assignment.grading_prompt,
    )

    return SubmissionSchema.model_validate(submission)


# ── Student: view own submission + grading result ─────────────────────────

@router.get("/{assignment_id}/my-submission", response_model=SubmissionDetailSchema)
async def get_my_submission(
    assignment_id: uuid.UUID,
    db:            AsyncSession = Depends(get_db),
    current_user:  User          = Depends(require_student),
):
    sub = await db.scalar(
        select(AssignmentSubmission).where(
            AssignmentSubmission.assignment_id == assignment_id,
            AssignmentSubmission.student_id    == current_user.id,
        )
    )
    if not sub:
        raise HTTPException(status_code=404, detail="No submission found for this assignment.")

    dl_url = storage.download_url(sub.cloudinary_id, sub.cloudinary_url)
    s = SubmissionDetailSchema.model_validate(sub)
    s.student_name  = current_user.full_name
    s.student_email = current_user.email
    s.file_download_url = dl_url
    return s
