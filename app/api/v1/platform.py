"""
v1 Platform API — lets external platforms (e.g. Learnify) use StudyMind's AI features.

Auth (see app/auth/api_key_auth.py):
- `Bearer sm_live_...` API key — server-to-server; course_id/user_id come from the request.
- `Bearer st_...` session token — browser-safe, minted via POST /auth/session and pinned to
  one course + user; course_id/user_id in the request are optional and must match if given.

Each (api_key, platform course, platform user) maps to a PlatformCourse row and a private
StudyMind Module owned by a synthetic platform user. Course content is indexed into that
module like any other class material, so chat/quiz/flashcards/summary reuse the
existing module-scoped pipelines.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.auth.api_key_auth import (
    PlatformIdentity, generate_session_token, get_api_key, get_platform_identity,
)
from app.db.engine import AsyncSessionLocal, get_db
from app.db.models import (
    ApiKey, Document, DocumentChunk, Module, ModuleDocument, PlatformCourse, SessionToken, User,
)
from app.ingestion.ingestor import get_ingestor
from app.logging_config import get_logger
from app.retrieval.typesense_client import get_typesense_client, mark_chunks_superseded

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Platform API"])

COURSE_CONTENT_FILENAME = "course_content.txt"


# ── Request schemas ────────────────────────────────────────────────────────

class LectureData(BaseModel):
    title:         str
    description:   Optional[str] = None
    resource_url:  Optional[str] = None  # e.g. PDF URL from Cloudinary (not fetched yet)
    resource_type: Optional[str] = None  # pdf | video | link


class SectionData(BaseModel):
    title:    str
    lectures: list[LectureData] = []


USER_ROLE_PATTERN = "^(student|tutor|admin)$"

# course_id / user_id are required with an API key; optional with a session token.
OptionalId = Optional[str]


class CreateSessionRequest(BaseModel):
    course_id:  str = Field(..., min_length=1, max_length=255)
    user_id:    str = Field(..., min_length=1, max_length=255)
    user_role:  str = Field(default="student", pattern=USER_ROLE_PATTERN)
    expires_in: int = Field(default=3600, ge=300, le=86400, description="Seconds (5 min – 24 h)")


class IngestCourseRequest(BaseModel):
    course_id:   OptionalId = Field(default=None, max_length=255, description="Platform's course ID")
    user_id:     OptionalId = Field(default=None, max_length=255, description="Platform's user ID")
    user_role:   str = Field(default="student", pattern=USER_ROLE_PATTERN)
    title:       str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    sections:    list[SectionData] = []


class ChatRequest(BaseModel):
    course_id:  OptionalId = None
    user_id:    OptionalId = None
    message:    str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    complexity: str = Field(default="normal", pattern="^(simple|normal|expert)$")


class QuizRequest(BaseModel):
    course_id:      OptionalId = None
    user_id:        OptionalId = None
    question_count: int = Field(default=5, ge=1, le=20)
    question_type:  str = Field(default="mcq", pattern="^(mcq|short_answer|true_false)$")
    topic:          Optional[str] = None


class FlashcardsRequest(BaseModel):
    course_id: OptionalId = None
    user_id:   OptionalId = None
    max_cards: int = Field(default=20, ge=5, le=50)
    topic:     Optional[str] = None


class SummaryRequest(BaseModel):
    course_id: OptionalId = None
    user_id:   OptionalId = None
    topic:     Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────

def platform_user_email(platform: str, user_id: str) -> str:
    return f"platform_{platform}_{user_id}@studymind.internal"


def build_course_text(req: IngestCourseRequest) -> str:
    """Flatten the platform's course structure into a Markdown-ish text document."""
    parts = [f"Course: {req.title}"]
    if req.description:
        parts.append(f"Description: {req.description}")
    parts.append("")

    for section in req.sections:
        parts.append(f"## {section.title}")
        for lecture in section.lectures:
            parts.append(f"### {lecture.title}")
            if lecture.description:
                parts.append(lecture.description)
            parts.append("")

    return "\n".join(parts)


async def _find_platform_course(
    db: AsyncSession, api_key_id: uuid.UUID, course_id: str, user_id: str,
) -> Optional[PlatformCourse]:
    result = await db.execute(
        select(PlatformCourse).where(
            PlatformCourse.api_key_id         == api_key_id,
            PlatformCourse.platform_course_id == course_id,
            PlatformCourse.platform_user_id   == user_id,
        )
    )
    return result.scalar_one_or_none()


async def _get_ready_course(
    db: AsyncSession, identity: PlatformIdentity, course_id: OptionalId, user_id: OptionalId,
) -> PlatformCourse:
    course_id, user_id = identity.scope(course_id, user_id)
    pc = await _find_platform_course(db, identity.api_key.id, course_id, user_id)
    if not pc or pc.status != "ready" or not pc.module_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Course not ready for AI. Status: {pc.status if pc else 'not_found'}. "
                "Call /api/v1/courses/ingest first."
            ),
        )
    return pc


async def get_or_create_module(
    db:          AsyncSession,
    api_key:     ApiKey,
    course_id:   str,
    user_id:     str,
    title:       str,
    description: Optional[str] = None,
) -> tuple[PlatformCourse, Module]:
    """Find or create the PlatformCourse and its backing StudyMind Module."""
    platform_course = await _find_platform_course(db, api_key.id, course_id, user_id)

    if platform_course and platform_course.module_id:
        module = await db.get(Module, platform_course.module_id)
        if module:
            # Keep metadata in sync with the platform
            platform_course.course_title       = title
            platform_course.course_description = description
            module.title       = title
            module.description = description
            await db.commit()
            return platform_course, module

    # Synthetic owner user for this platform user
    email  = platform_user_email(api_key.platform, user_id)
    result = await db.execute(select(User).where(User.email == email))
    platform_user = result.scalar_one_or_none()

    if not platform_user:
        platform_user = User(
            id=uuid.uuid4(),
            email=email,
            full_name=f"{api_key.platform} user {user_id}"[:255],
            role="student",
            password_hash="!",  # unusable — platform users never log in directly
            is_active=True,
            is_verified=True,
        )
        db.add(platform_user)
        await db.flush()

    module = Module(
        id=uuid.uuid4(),
        title=title,
        description=description,
        course_code=f"{api_key.platform[:3]}{course_id[:6]}".upper(),
        owner_id=platform_user.id,
        access_type="personal",
        status="active",
        module_metadata={
            "source":             "platform",
            "platform":           api_key.platform,
            "platform_course_id": course_id,
        },
    )
    db.add(module)
    await db.flush()

    if platform_course:
        platform_course.module_id = module.id
    else:
        platform_course = PlatformCourse(
            id=uuid.uuid4(),
            api_key_id=api_key.id,
            platform_course_id=course_id,
            platform_user_id=user_id,
            module_id=module.id,
            course_title=title,
            course_description=description,
            status="pending",
        )
        db.add(platform_course)

    await db.commit()
    return platform_course, module


async def index_course_content(platform_course_id: uuid.UUID, course_text: str) -> None:
    """
    Background task: index course text into the course's module.
    Uses its own DB session — the request session is closed by the time this runs.
    Re-ingesting replaces the previous version's chunks (MOD-04 versioning).
    """
    async with AsyncSessionLocal() as db:
        pc = await db.get(PlatformCourse, platform_course_id)
        if not pc or not pc.module_id:
            return
        module = await db.get(Module, pc.module_id)
        if not module:
            pc.status        = "failed"
            pc.error_message = "Module missing"
            await db.commit()
            return

        try:
            # Supersede the previous version of the course content, if any
            prev_result = await db.execute(
                select(ModuleDocument)
                .join(Document, Document.id == ModuleDocument.document_id)
                .where(
                    ModuleDocument.module_id == module.id,
                    Document.filename == COURSE_CONTENT_FILENAME,
                    ModuleDocument.is_latest == True,  # noqa: E712
                )
            )
            prev_md     = prev_result.scalars().first()
            new_version = 1
            if prev_md:
                new_version       = prev_md.version + 1
                prev_md.is_latest = False
                await run_in_threadpool(
                    mark_chunks_superseded,
                    get_typesense_client(), str(prev_md.document_id), prev_md.version,
                )

            content = course_text.encode("utf-8")
            doc = Document(
                id=uuid.uuid4(),
                owner_id=module.owner_id,
                filename=COURSE_CONTENT_FILENAME,
                file_type="txt",
                file_size_bytes=len(content),
                visibility="class",
                status="pending",
                doc_metadata={"source": "platform", "platform_course_id": pc.platform_course_id},
            )
            db.add(doc)
            await db.commit()

            ingestor = get_ingestor()
            result   = await run_in_threadpool(
                ingestor.ingest,
                COURSE_CONTENT_FILENAME,
                content,
                str(doc.id),
                module_id=str(module.id),
                course_code=module.course_code or "",
                visibility="class",
                doc_version=new_version,
                lecturer_id=str(module.owner_id),
            )

            for chunk in result["chunks"]:
                db.add(DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    typesense_id=chunk["typesense_id"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    token_count=chunk["token_count"],
                    chunk_metadata=chunk["chunk_metadata"],
                ))
            doc.status        = result["status"]
            doc.chunk_count   = result["chunk_count"]
            doc.error_message = result["error"]
            doc.indexed_at    = result["indexed_at"]

            db.add(ModuleDocument(
                id=uuid.uuid4(),
                module_id=module.id,
                document_id=doc.id,
                uploaded_by=module.owner_id,
                version=new_version,
                is_latest=True,
                visibility="class",
            ))

            if result["status"] == "indexed":
                pc.status        = "ready"
                pc.indexed_at    = datetime.now(timezone.utc)
                pc.chunk_count   = result["chunk_count"]
                pc.error_message = None
            else:
                pc.status        = "failed"
                pc.error_message = result["error"]
            await db.commit()

            log.info(
                "platform_course_indexed",
                platform_course_id=str(pc.id),
                status=pc.status,
                chunks=result["chunk_count"],
                version=new_version,
            )

        except Exception as exc:
            log.error("platform_course_index_failed", platform_course_id=str(platform_course_id), error=str(exc))
            await db.rollback()
            pc = await db.get(PlatformCourse, platform_course_id)
            if pc:
                pc.status        = "failed"
                pc.error_message = str(exc)[:1000]
                await db.commit()


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/auth/session", status_code=201)
async def create_session(
    req:     CreateSessionRequest,
    db:      AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(get_api_key),  # API key only — session tokens can't mint tokens
):
    """
    Exchange an API key for a short-lived session token scoped to one course + user.
    Call this SERVER-SIDE in the host app and pass only the session token to the browser.
    """
    pc = await _find_platform_course(db, api_key.id, req.course_id, req.user_id)

    full_token, token_hash = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=req.expires_in)

    db.add(SessionToken(
        id=uuid.uuid4(),
        token_hash=token_hash,
        api_key_id=api_key.id,
        course_id=req.course_id,
        user_id=req.user_id,
        user_role=req.user_role,
        module_id=pc.module_id if pc else None,
        expires_at=expires_at,
    ))
    await db.commit()

    return {
        "session_token": full_token,
        "expires_in":    req.expires_in,
        "expires_at":    expires_at.isoformat(),
        "course_id":     req.course_id,
        "user_id":       req.user_id,
        "course_status": pc.status if pc else "not_found",
    }


@router.post("/courses/ingest", status_code=202)
async def ingest_course(
    req:              IngestCourseRequest,
    background_tasks: BackgroundTasks,
    db:               AsyncSession = Depends(get_db),
    identity:         PlatformIdentity = Depends(get_platform_identity),
):
    """
    Ingest (or re-ingest) a platform course. Creates a StudyMind module and indexes
    the course content in the background. Poll /api/v1/courses/status for progress.
    With a session token, only the token's own course can be ingested.
    """
    course_id, user_id = identity.scope(req.course_id, req.user_id)
    platform_course, module = await get_or_create_module(
        db, identity.api_key, course_id, user_id, req.title, req.description,
    )

    platform_course.status = "indexing"
    await db.commit()

    background_tasks.add_task(index_course_content, platform_course.id, build_course_text(req))

    return {
        "module_id": str(module.id),
        "course_id": course_id,
        "status":    "indexing",
        "message":   "Course content is being indexed. Use /api/v1/courses/status to check.",
    }


@router.get("/courses/status")
async def get_course_status(
    course_id: OptionalId = None,
    user_id:   OptionalId = None,
    db:        AsyncSession = Depends(get_db),
    identity:  PlatformIdentity = Depends(get_platform_identity),
):
    """Check indexing status of a course."""
    course_id, user_id = identity.scope(course_id, user_id)
    pc = await _find_platform_course(db, identity.api_key.id, course_id, user_id)
    if not pc:
        return {"status": "not_found", "message": "Course not ingested yet"}

    return {
        "status":      pc.status,
        "module_id":   str(pc.module_id) if pc.module_id else None,
        "chunk_count": pc.chunk_count,
        "indexed_at":  pc.indexed_at.isoformat() if pc.indexed_at else None,
        "error":       pc.error_message if pc.status == "failed" else None,
    }


@router.post("/chat")
async def platform_chat(
    req:      ChatRequest,
    db:       AsyncSession = Depends(get_db),
    identity: PlatformIdentity = Depends(get_platform_identity),
):
    """AI Tutor — answer a question scoped to a platform course."""
    pc = await _get_ready_course(db, identity, req.course_id, req.user_id)

    from app.agents.rag_pipeline import get_pipeline
    result = await run_in_threadpool(
        get_pipeline().query,
        question=req.message,
        module_id=str(pc.module_id),
        scope_mode="everything",
        complexity=req.complexity,
    )

    return {
        "answer":           result["answer"],
        "sources":          result["sources"],
        "no_content_found": result["no_content_found"],
        # Echoed back; conversation history is not persisted for platform chats yet
        "session_id":       req.session_id,
    }


@router.post("/quiz/generate")
async def platform_generate_quiz(
    req:      QuizRequest,
    db:       AsyncSession = Depends(get_db),
    identity: PlatformIdentity = Depends(get_platform_identity),
):
    """Generate a quiz for a platform course."""
    pc = await _get_ready_course(db, identity, req.course_id, req.user_id)

    from app.agents.ai_features import generate_quiz
    try:
        questions = await run_in_threadpool(
            generate_quiz,
            module_id=str(pc.module_id),
            question_count=req.question_count,
            question_type=req.question_type,
            title=pc.course_title,
            topic=req.topic,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"course_id": pc.platform_course_id, "questions": questions, "count": len(questions)}


@router.post("/flashcards/generate")
async def platform_generate_flashcards(
    req:      FlashcardsRequest,
    db:       AsyncSession = Depends(get_db),
    identity: PlatformIdentity = Depends(get_platform_identity),
):
    """Generate flashcards for a platform course."""
    pc = await _get_ready_course(db, identity, req.course_id, req.user_id)

    from app.agents.ai_features import generate_flashcards
    try:
        cards = await run_in_threadpool(
            generate_flashcards,
            module_id=str(pc.module_id),
            max_cards=req.max_cards,
            topic=req.topic,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"course_id": pc.platform_course_id, "cards": cards, "count": len(cards)}


@router.post("/summarise")
async def platform_summarise(
    req:      SummaryRequest,
    db:       AsyncSession = Depends(get_db),
    identity: PlatformIdentity = Depends(get_platform_identity),
):
    """Generate a Markdown summary for a platform course."""
    pc = await _get_ready_course(db, identity, req.course_id, req.user_id)

    from app.agents.ai_features import generate_summary
    try:
        summary, _chunk_count = await run_in_threadpool(
            generate_summary,
            module_id=str(pc.module_id),
            scope="module",
            topic=req.topic,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"course_id": pc.platform_course_id, "summary": summary}
