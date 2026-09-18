"""
Modules, Semesters, Weeks & Enrolment endpoints — Sprint 3.

MOD-01: POST /api/modules                    — create module (lecturer/self-learner)
MOD-02: Semester scoping                     — automatic via semester_id
MOD-03: POST /api/modules/{id}/weeks         — add week/topic
MOD-06: Self-learner creates personal course — same endpoint, access_type=personal
MOD-07: PATCH /api/modules/{id}/status       — active | archived | draft
INST-06: POST /api/modules/{id}/enrol        — student enrols via code or direct
DOC-01-03: POST /api/modules/{id}/documents  — upload with full scoping
DOC-10: DELETE /api/modules/{id}/documents/{doc_id}
DOC-11: GET /api/modules/{id}/documents/{doc_id}/versions
"""
from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth.dependencies import require_auth, require_lecturer, require_student
from app.config import get_settings
from app.db.engine import get_db
from app.db.models import (
    Document, DocumentChunk, Enrolment, InstitutionCode,
    Module, ModuleDocument, Semester, User, Week,
)
from app.db.schemas import (
    DocumentUploadResponse, DocumentVersionSchema,
    EnrolRequest, EnrolmentSchema,
    ModuleCreate, ModuleDetailSchema, ModuleDocumentSchema, ModuleSchema,
    SemesterCreate, SemesterSchema,
    WeekCreate, WeekSchema,
)
from app.ingestion.ingestor import ALLOWED_EXTENSIONS, get_ingestor
from app.logging_config import get_logger
from app.retrieval.typesense_client import (
    delete_document_chunks, get_typesense_client, mark_chunks_superseded,
)

log      = get_logger(__name__)
router   = APIRouter(prefix="/api", tags=["modules"])
executor = ThreadPoolExecutor(max_workers=4)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ═══════════════════════════════════════════════════════════════════════════
# Semesters
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/semesters", response_model=SemesterSchema, status_code=201)
async def create_semester(
    body: SemesterCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_lecturer),
):
    """Create a new semester for the current user's institution."""
    # If marking as current, un-current all others for this institution
    if body.is_current and current_user.institution_id:
        existing = await db.execute(
            select(Semester).where(
                Semester.institution_id == current_user.institution_id,
                Semester.is_current == True,
            )
        )
        for s in existing.scalars().all():
            s.is_current = False

    semester = Semester(
        id=uuid.uuid4(),
        institution_id=current_user.institution_id,
        label=body.label,
        start_date=body.start_date,
        end_date=body.end_date,
        is_current=body.is_current,
    )
    db.add(semester)
    await db.commit()
    await db.refresh(semester)
    log.info("semester_created", label=semester.label)
    return semester


@router.get("/semesters", response_model=list[SemesterSchema])
async def list_semesters(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    result = await db.execute(
        select(Semester)
        .where(Semester.institution_id == current_user.institution_id)
        .order_by(Semester.created_at.desc())
    )
    return result.scalars().all()


# ═══════════════════════════════════════════════════════════════════════════
# Modules — MOD-01, MOD-06, MOD-07
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/modules", response_model=ModuleSchema, status_code=201)
async def create_module(
    body: ModuleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    MOD-01: Lecturer creates a class module.
    MOD-06: Self-learner creates a personal course (access_type=personal).
    """
    # Self-learners can only create personal courses
    if current_user.role == "self_learner" and body.access_type != "personal":
        raise HTTPException(
            status_code=403,
            detail="Self-learners can only create personal courses (access_type=personal)",
        )

    module = Module(
        id=uuid.uuid4(),
        course_code=body.course_code.upper() if body.course_code else None,
        title=body.title,
        description=body.description,
        owner_id=current_user.id,
        institution_id=current_user.institution_id,
        department_id=body.department_id or current_user.department_id,
        semester_id=body.semester_id,
        access_type=body.access_type,
        status=body.status,
    )
    db.add(module)
    await db.commit()
    await db.refresh(module)

    schema                = ModuleSchema.model_validate(module)
    schema.document_count = 0
    schema.student_count  = 0
    log.info("module_created", module_id=str(module.id), course_code=module.course_code)
    return schema


@router.get("/modules", response_model=list[ModuleSchema])
async def list_modules(
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    Lecturer: returns modules they own.
    Student: returns modules they are enrolled in.
    Self-learner: returns their personal courses.
    """
    if current_user.role in ("lecturer", "admin"):
        query = select(Module).where(Module.owner_id == current_user.id)
    elif current_user.role == "student":
        query = (
            select(Module)
            .join(Enrolment, Enrolment.module_id == Module.id)
            .where(Enrolment.student_id == current_user.id,
                   Enrolment.status == "active")
        )
    else:  # self_learner
        query = select(Module).where(Module.owner_id == current_user.id)

    if status:
        query = query.where(Module.status == status)

    result  = await db.execute(query.order_by(Module.updated_at.desc()))
    modules = result.scalars().all()

    out = []
    for m in modules:
        doc_count = (await db.execute(
            select(func.count()).select_from(ModuleDocument)
            .where(ModuleDocument.module_id == m.id, ModuleDocument.is_latest == True)
        )).scalar() or 0
        student_count = (await db.execute(
            select(func.count()).select_from(Enrolment)
            .where(Enrolment.module_id == m.id, Enrolment.status == "active")
        )).scalar() or 0
        s                = ModuleSchema.model_validate(m)
        s.document_count = doc_count
        s.student_count  = student_count
        out.append(s)
    return out


@router.get("/modules/{module_id}", response_model=ModuleDetailSchema)
async def get_module(
    module_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    module = await _get_accessible_module(module_id, current_user, db)
    result = await db.execute(
        select(Week).where(Week.module_id == module_id).order_by(Week.week_number)
    )
    weeks  = result.scalars().all()

    schema          = ModuleDetailSchema.model_validate(module)
    schema.weeks    = [WeekSchema.model_validate(w) for w in weeks]
    schema.document_count = (await db.execute(
        select(func.count()).select_from(ModuleDocument)
        .where(ModuleDocument.module_id == module_id, ModuleDocument.is_latest == True)
    )).scalar() or 0
    schema.student_count = (await db.execute(
        select(func.count()).select_from(Enrolment)
        .where(Enrolment.module_id == module_id, Enrolment.status == "active")
    )).scalar() or 0
    return schema


@router.patch("/modules/{module_id}", response_model=ModuleSchema)
async def update_module(
    module_id: uuid.UUID,
    body: ModuleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    module = await _get_owned_module(module_id, current_user, db)
    module.title        = body.title
    module.course_code  = body.course_code.upper() if body.course_code else None
    module.description  = body.description
    module.access_type  = body.access_type
    module.status       = body.status
    module.semester_id  = body.semester_id
    await db.commit()
    await db.refresh(module)
    schema = ModuleSchema.model_validate(module)
    schema.document_count = 0
    schema.student_count  = 0
    return schema


@router.patch("/modules/{module_id}/status", response_model=ModuleSchema)
async def update_module_status(
    module_id: uuid.UUID,
    new_status: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """MOD-07: Set module status — active | archived | draft."""
    if new_status not in {"active", "archived", "draft"}:
        raise HTTPException(status_code=422, detail="status must be: active | archived | draft")
    module        = await _get_owned_module(module_id, current_user, db)
    module.status = new_status
    await db.commit()
    await db.refresh(module)
    schema = ModuleSchema.model_validate(module)
    schema.document_count = 0
    schema.student_count  = 0
    return schema


@router.delete("/modules/{module_id}", status_code=204)
async def delete_module(
    module_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    module = await _get_owned_module(module_id, current_user, db)
    await db.delete(module)
    await db.commit()


# ═══════════════════════════════════════════════════════════════════════════
# Weeks — MOD-03
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/modules/{module_id}/weeks", response_model=WeekSchema, status_code=201)
async def create_week(
    module_id: uuid.UUID,
    body: WeekCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """MOD-03: Add a week / topic to a module."""
    await _get_owned_module(module_id, current_user, db)

    # Check week number not already taken
    existing = await db.execute(
        select(Week).where(Week.module_id == module_id, Week.week_number == body.week_number)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail=f"Week {body.week_number} already exists in this module",
        )

    week = Week(
        id=uuid.uuid4(),
        module_id=module_id,
        week_number=body.week_number,
        title=body.title,
        description=body.description,
    )
    db.add(week)
    await db.commit()
    await db.refresh(week)
    return week


@router.get("/modules/{module_id}/weeks", response_model=list[WeekSchema])
async def list_weeks(
    module_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    await _get_accessible_module(module_id, current_user, db)
    result = await db.execute(
        select(Week).where(Week.module_id == module_id).order_by(Week.week_number)
    )
    return result.scalars().all()


@router.patch("/modules/{module_id}/weeks/{week_id}", response_model=WeekSchema)
async def update_week(
    module_id: uuid.UUID,
    week_id: uuid.UUID,
    body: WeekCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    await _get_owned_module(module_id, current_user, db)
    result = await db.execute(
        select(Week).where(Week.id == week_id, Week.module_id == module_id)
    )
    week = result.scalar_one_or_none()
    if not week:
        raise HTTPException(status_code=404, detail="Week not found")
    week.title       = body.title
    week.description = body.description
    await db.commit()
    await db.refresh(week)
    return week


@router.delete("/modules/{module_id}/weeks/{week_id}", status_code=204)
async def delete_week(
    module_id: uuid.UUID,
    week_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    await _get_owned_module(module_id, current_user, db)
    result = await db.execute(
        select(Week).where(Week.id == week_id, Week.module_id == module_id)
    )
    week = result.scalar_one_or_none()
    if not week:
        raise HTTPException(status_code=404, detail="Week not found")
    await db.delete(week)
    await db.commit()


# ═══════════════════════════════════════════════════════════════════════════
# Enrolment — INST-05, INST-06
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/modules/{module_id}/enrol", response_model=EnrolmentSchema, status_code=201)
async def enrol_in_module(
    module_id: uuid.UUID,
    body: EnrolRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """Student enrols in a module via enrolment code or direct module ID."""
    if current_user.role not in ("student", "self_learner"):
        raise HTTPException(status_code=403, detail="Only students can enrol in modules")

    # Resolve module — either direct or via enrolment code
    if body.enrolment_code:
        code_result = await db.execute(
            select(InstitutionCode).where(
                InstitutionCode.code == body.enrolment_code,
                InstitutionCode.code_type == "module_enrolment",
                InstitutionCode.is_active == True,
            )
        )
        code = code_result.scalar_one_or_none()
        if not code:
            raise HTTPException(status_code=404, detail="Invalid enrolment code")
        if code.expires_at and code.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            raise HTTPException(status_code=410, detail="Enrolment code has expired")
        if code.max_uses and code.use_count >= code.max_uses:
            raise HTTPException(status_code=410, detail="Enrolment code has reached its limit")
        # Get module from code metadata
        module_id_from_code = (code.metadata_ or {}).get("module_id")
        if not module_id_from_code:
            raise HTTPException(status_code=422, detail="Code is not linked to a module")
        module_id = uuid.UUID(module_id_from_code)
        code.use_count += 1

    # Check module exists and is enrollable
    module = await _get_module_or_404(module_id, db)
    if module.access_type == "personal" and module.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot enrol in a personal module")
    if module.status == "draft":
        raise HTTPException(status_code=403, detail="Module is not yet open for enrolment")

    # Check not already enrolled
    existing = await db.execute(
        select(Enrolment).where(
            Enrolment.student_id == current_user.id,
            Enrolment.module_id == module_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Already enrolled in this module")

    enrolment = Enrolment(
        id=uuid.uuid4(),
        student_id=current_user.id,
        module_id=module_id,
        status="active",
    )
    db.add(enrolment)
    await db.commit()
    await db.refresh(enrolment)
    log.info("student_enrolled", student_id=str(current_user.id), module_id=str(module_id))
    return enrolment


@router.get("/modules/{module_id}/students", response_model=list[dict])
async def list_enrolled_students(
    module_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_lecturer),
):
    """Lecturer views all enrolled students in their module."""
    await _get_owned_module(module_id, current_user, db)
    result = await db.execute(
        select(Enrolment, User)
        .join(User, User.id == Enrolment.student_id)
        .where(Enrolment.module_id == module_id, Enrolment.status == "active")
        .order_by(User.full_name)
    )
    return [
        {
            "enrolment_id": str(e.id),
            "student_id":   str(u.id),
            "full_name":    u.full_name,
            "email":        u.email,
            "enrolled_at":  e.enrolled_at.isoformat(),
        }
        for e, u in result.all()
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Document Upload — DOC-01, DOC-02, DOC-03, DOC-04
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/modules/{module_id}/documents", response_model=DocumentUploadResponse, status_code=201)
async def upload_module_document(
    module_id: uuid.UUID,
    file: UploadFile = File(...),
    week_id: uuid.UUID | None = None,
    visibility: str = "class",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    DOC-01: Lecturer uploads class material.
    DOC-02: Student uploads personal notes (visibility=personal).
    DOC-03: Self-learner uploads to personal course.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    module = await _get_accessible_module(module_id, current_user, db)

    # Access control: only lecturer/owner can upload class materials
    if visibility == "class" and current_user.role == "student":
        raise HTTPException(
            status_code=403,
            detail="Students cannot upload class materials. Use visibility=personal for personal notes.",
        )

    # Resolve week metadata
    week_number = 0
    week_id_str = ""
    if week_id:
        week_result = await db.execute(
            select(Week).where(Week.id == week_id, Week.module_id == module_id)
        )
        week = week_result.scalar_one_or_none()
        if not week:
            raise HTTPException(status_code=404, detail="Week not found in this module")
        week_number = week.week_number
        week_id_str = str(week_id)

    # Resolve semester metadata
    semester_id_str     = ""
    is_current_semester = True
    if module.semester_id:
        sem_result = await db.execute(select(Semester).where(Semester.id == module.semester_id))
        sem = sem_result.scalar_one_or_none()
        if sem:
            semester_id_str     = str(sem.id)
            is_current_semester = sem.is_current

    # Determine version — MOD-04
    existing_docs = await db.execute(
        select(ModuleDocument)
        .join(Document, Document.id == ModuleDocument.document_id)
        .where(
            ModuleDocument.module_id == module_id,
            Document.filename == file.filename,
            ModuleDocument.is_latest == True,
        )
    )
    existing_md = existing_docs.scalar_one_or_none()
    new_version = 1

    if existing_md:
        # MOD-04: New version — mark old as superseded
        new_version          = existing_md.version + 1
        existing_md.is_latest = False
        # Mark old Typesense chunks as superseded
        old_doc_id = str(existing_md.document_id)
        ts_client  = get_typesense_client()
        mark_chunks_superseded(ts_client, old_doc_id, existing_md.version)
        log.info("document_new_version", filename=file.filename, new_version=new_version)

    # Create Document record
    doc = Document(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        filename=file.filename,
        file_type=ext.lstrip("."),
        file_size_bytes=len(content),
        visibility=visibility,
        status="pending",
    )
    db.add(doc)
    await db.flush()
    await db.commit()

    doc_id = str(doc.id)

    # Build scoping params for ingestor
    ingest_params = dict(
        module_id=str(module_id),
        course_code=module.course_code or "",
        semester_id=semester_id_str,
        week_id=week_id_str,
        week_number=week_number,
        visibility=visibility,
        owner_id=str(current_user.id) if visibility == "personal" else "",
        doc_version=new_version,
        is_current_semester=is_current_semester,
        institution_id=str(module.institution_id) if module.institution_id else "",
        lecturer_id=str(module.owner_id),
    )

    # Run ingestion in thread pool
    ingestor = get_ingestor()
    settings = get_settings()
    loop     = asyncio.get_event_loop()

    def _run():
        try:
            r = ingestor.ingest(file.filename, content, doc_id, **ingest_params)
            print(f"INGEST RESULT: {r['status']} chunks={r['chunk_count']} error={r['error']}")
            return r
        except Exception as e:
            print(f"INGEST EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            raise

    result = await loop.run_in_executor(executor, _run)

    # Persist chunks and update document status
    def _persist(doc_id_str: str, result: dict, new_version: int):
        engine = create_engine(settings.postgres_dsn_sync)
        with Session(engine) as session:
            doc_obj = session.get(Document, uuid.UUID(doc_id_str))
            if not doc_obj:
                return
            for chunk in result["chunks"]:
                session.add(DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc_obj.id,
                    typesense_id=chunk["typesense_id"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    token_count=chunk["token_count"],
                    chunk_metadata=chunk["chunk_metadata"],
                ))
            doc_obj.status      = result["status"]
            doc_obj.chunk_count = result["chunk_count"]
            doc_obj.error_message = result["error"]
            doc_obj.indexed_at  = result["indexed_at"]
            session.commit()
        engine.dispose()

    await loop.run_in_executor(executor, _persist, doc_id, result, new_version)

    # Create ModuleDocument link
    md = ModuleDocument(
        id=uuid.uuid4(),
        module_id=module_id,
        document_id=doc.id,
        week_id=week_id,
        uploaded_by=current_user.id,
        version=new_version,
        is_latest=True,
        visibility=visibility,
    )
    db.add(md)
    await db.commit()

    log.info(
        "module_document_uploaded",
        module_id=str(module_id),
        filename=file.filename,
        version=new_version,
        status=result["status"],
        chunks=result["chunk_count"],
    )

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=file.filename,
        status=result["status"],
        message=(
            f"Indexed {result['chunk_count']} chunks from '{file.filename}' (v{new_version})"
            if result["status"] == "indexed"
            else f"Ingestion failed: {result['error']}"
        ),
    )


@router.get("/modules/{module_id}/documents", response_model=list[ModuleDocumentSchema])
async def list_module_documents(
    module_id: uuid.UUID,
    week_id: uuid.UUID | None = None,
    latest_only: bool = True,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """List documents in a module. Students only see class + their own personal."""
    await _get_accessible_module(module_id, current_user, db)

    query = (
        select(ModuleDocument, Document)
        .join(Document, Document.id == ModuleDocument.document_id)
        .where(ModuleDocument.module_id == module_id)
    )

    if latest_only:
        query = query.where(ModuleDocument.is_latest == True)
    if week_id:
        query = query.where(ModuleDocument.week_id == week_id)

    # SRCH-08: Students see class + their own personal notes only
    if current_user.role == "student":
        query = query.where(
            (ModuleDocument.visibility == "class") |
            ((ModuleDocument.visibility == "personal") & (Document.owner_id == current_user.id))
        )

    result = await db.execute(query.order_by(ModuleDocument.created_at.desc()))
    out    = []
    for md, doc in result.all():
        s                   = ModuleDocumentSchema.model_validate(md)
        s.filename          = doc.filename
        s.file_type         = doc.file_type
        s.file_size_bytes   = doc.file_size_bytes
        s.chunk_count       = doc.chunk_count
        s.status            = doc.status
        s.indexed_at        = doc.indexed_at
        out.append(s)
    return out


@router.delete("/modules/{module_id}/documents/{document_id}", status_code=204)
async def delete_module_document(
    module_id: uuid.UUID,
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """DOC-10: Delete a document from a module."""
    await _get_accessible_module(module_id, current_user, db)

    result = await db.execute(
        select(ModuleDocument, Document)
        .join(Document, Document.id == ModuleDocument.document_id)
        .where(
            ModuleDocument.module_id == module_id,
            ModuleDocument.document_id == document_id,
        )
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found in this module")

    md, doc = row

    # Only lecturer/owner or the personal note owner can delete
    if current_user.role == "student" and doc.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="You can only delete your own personal notes")

    ts_client = get_typesense_client()
    delete_document_chunks(ts_client, str(document_id))

    await db.delete(md)
    await db.delete(doc)
    await db.commit()
    log.info("module_document_deleted", document_id=str(document_id))


@router.get(
    "/modules/{module_id}/documents/{document_id}/versions",
    response_model=list[DocumentVersionSchema],
)
async def get_document_versions(
    module_id: uuid.UUID,
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """DOC-11: Show version history for a document."""
    await _get_accessible_module(module_id, current_user, db)

    result = await db.execute(
        select(ModuleDocument, Document)
        .join(Document, Document.id == ModuleDocument.document_id)
        .where(ModuleDocument.module_id == module_id)
        .order_by(ModuleDocument.version.desc())
    )

    # Find the filename of this document first
    target = await db.execute(select(Document).where(Document.id == document_id))
    target_doc = target.scalar_one_or_none()
    if not target_doc:
        raise HTTPException(status_code=404, detail="Document not found")

    versions = []
    for md, doc in result.all():
        if doc.filename == target_doc.filename:
            versions.append(DocumentVersionSchema(
                version=md.version,
                created_at=md.created_at,
                is_latest=md.is_latest,
                document_id=doc.id,
                filename=doc.filename,
            ))
    return versions


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

async def _get_module_or_404(module_id: uuid.UUID, db: AsyncSession) -> Module:
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return module


async def _get_owned_module(
    module_id: uuid.UUID, user: User, db: AsyncSession
) -> Module:
    """Module the current user owns — for write operations."""
    module = await _get_module_or_404(module_id, db)
    if module.owner_id != user.id and user.role != "admin":
        raise HTTPException(status_code=403, detail="You do not own this module")
    return module


async def _get_accessible_module(
    module_id: uuid.UUID, user: User, db: AsyncSession
) -> Module:
    """Module the current user can read — owner OR enrolled student."""
    module = await _get_module_or_404(module_id, db)

    # Owner always has access
    if module.owner_id == user.id or user.role == "admin":
        return module

    # Institution-wide module
    if module.access_type == "institution" and module.institution_id == user.institution_id:
        return module

    # Student must be enrolled
    if user.role == "student":
        enrolment = await db.execute(
            select(Enrolment).where(
                Enrolment.student_id == user.id,
                Enrolment.module_id == module_id,
                Enrolment.status == "active",
            )
        )
        if enrolment.scalar_one_or_none():
            return module

    raise HTTPException(status_code=403, detail="You do not have access to this module")
