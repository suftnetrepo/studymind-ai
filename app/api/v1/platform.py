"""
v1 Platform API — lets external platforms (e.g. Learnify) use StudyMind's AI features.

Auth (see app/auth/api_key_auth.py):
- `Bearer sm_live_...` API key — server-to-server; course_id/user_id come from the request.
- `Bearer st_...` session token — browser-safe, minted via POST /auth/session and pinned to
  one course + user; course_id/user_id in the request are optional and must match if given.

Each (api_key, platform course) has ONE shared StudyMind Module; every platform user of that
course gets a PlatformCourse row (per-user status) pointing at it. Course content and uploaded
materials (see documents.py) are indexed into that module as class material, so tutors'
uploads reach every student and chat/quiz/flashcards/summary reuse the module-scoped pipelines.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.auth.api_key_auth import (
    PlatformIdentity, generate_session_token, get_api_key, get_platform_identity,
)
from app.db.engine import AsyncSessionLocal, get_db
from app.db.models import (
    ApiKey, ChatMessage, ChatSession, Document, DocumentChunk, Module, ModuleDocument, PlatformCourse,
    PlatformDocument, PlatformSummary, SessionToken, User,
)
from app.ingestion.ingestor import get_ingestor
from app.logging_config import get_logger
from app.retrieval.typesense_client import delete_document_chunks, get_typesense_client

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Platform API"])

COURSE_CONTENT_FILENAME = "course_content.txt"
CHAT_HISTORY_TURNS      = 10   # previous Q&A pairs sent to the model as context


# ── Request schemas ────────────────────────────────────────────────────────

class LectureData(BaseModel):
    title:         str
    description:   Optional[str] = None
    resource_url:  Optional[str] = None  # e.g. PDF URL from Cloudinary (not fetched yet)
    resource_type: Optional[str] = None  # pdf | video | link


class SectionData(BaseModel):
    title:    str
    lectures: list[LectureData] = []


USER_ROLE_PATTERN  = "^(student|tutor|admin)$"
COMPLEXITY_PATTERN = "^(simple|normal|expert)$"

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
    complexity: str = Field(default="normal", pattern=COMPLEXITY_PATTERN)


class QuizRequest(BaseModel):
    course_id:      OptionalId = None
    user_id:        OptionalId = None
    question_count: int = Field(default=5, ge=1, le=20)
    question_type:  str = Field(default="mcq", pattern="^(mcq|short_answer|true_false)$")
    topic:          Optional[str] = None
    complexity:     str = Field(default="normal", pattern=COMPLEXITY_PATTERN)


class FlashcardsRequest(BaseModel):
    course_id: OptionalId = None
    user_id:   OptionalId = None
    max_cards: int = Field(default=20, ge=5, le=50)
    topic:     Optional[str] = None
    complexity: str = Field(default="normal", pattern=COMPLEXITY_PATTERN)


class SummaryRequest(BaseModel):
    course_id: OptionalId = None
    user_id:   OptionalId = None
    topic:     Optional[str] = None
    complexity: str = Field(default="normal", pattern=COMPLEXITY_PATTERN)


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


@dataclass
class ReadyCourse:
    """What the AI endpoints need: the course and the module to search."""
    platform_course_id: str
    platform_user_id:   str
    module_id:          uuid.UUID
    course_title:       str


async def _get_ready_course(
    db: AsyncSession, identity: PlatformIdentity, course_id: OptionalId, user_id: OptionalId,
) -> ReadyCourse:
    """
    A course is ready for AI when this user's course content is indexed, OR when the course has
    at least one indexed uploaded document (platforms may upload materials without ever calling
    /courses/ingest, and students need no PlatformCourse row of their own to use them).
    """
    course_id, user_id = identity.scope(course_id, user_id)
    pc = await _find_platform_course(db, identity.api_key.id, course_id, user_id)
    if pc and pc.status == "ready" and pc.module_id:
        return ReadyCourse(course_id, user_id, pc.module_id, pc.course_title)

    doc_result = await db.execute(
        select(PlatformDocument)
        .where(
            PlatformDocument.api_key_id         == identity.api_key.id,
            PlatformDocument.platform_course_id == course_id,
            PlatformDocument.status             == "ready",
            PlatformDocument.module_id.is_not(None),
        )
        .limit(1)
    )
    ready_doc = doc_result.scalars().first()
    if ready_doc:
        title = pc.course_title if pc else f"Course {course_id}"
        return ReadyCourse(course_id, user_id, ready_doc.module_id, title)

    raise HTTPException(
        status_code=400,
        detail=(
            f"Course not ready for AI. Status: {pc.status if pc else 'not_found'}. "
            "Call /api/v1/courses/ingest or upload course materials first."
        ),
    )


async def _xact_lock(db: AsyncSession, key: str) -> None:
    """
    Transaction-scoped Postgres advisory lock (released on commit/rollback). Serialises
    concurrent first-time creation of the same course/user — e.g. a panel's parallel initial
    requests — which would otherwise race on unique constraints or split a course across modules.
    """
    await db.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": key})


async def _shared_module(db: AsyncSession, api_key_id: uuid.UUID, course_id: str) -> Optional[Module]:
    """The course's shared module: the one linked from its earliest PlatformCourse row."""
    result = await db.execute(
        select(Module)
        .join(PlatformCourse, PlatformCourse.module_id == Module.id)
        .where(
            PlatformCourse.api_key_id         == api_key_id,
            PlatformCourse.platform_course_id == course_id,
        )
        .order_by(PlatformCourse.created_at)
        .limit(1)
    )
    return result.scalars().first()


async def _platform_user(db: AsyncSession, platform: str, user_id: str) -> User:
    """Synthetic StudyMind user for a platform user (owns modules/documents; never logs in)."""
    email  = platform_user_email(platform, user_id)
    await _xact_lock(db, f"platform_user:{email}")
    result = await db.execute(select(User).where(User.email == email))
    user   = result.scalar_one_or_none()
    if not user:
        user = User(
            id=uuid.uuid4(),
            email=email,
            full_name=f"{platform} user {user_id}"[:255],
            role="student",
            password_hash="!",  # unusable
            is_active=True,
            is_verified=True,
        )
        db.add(user)
        await db.flush()
    return user


async def get_or_create_module(
    db:          AsyncSession,
    api_key:     ApiKey,
    course_id:   str,
    user_id:     str,
    title:       Optional[str] = None,
    description: Optional[str] = None,
) -> tuple[PlatformCourse, Module]:
    """
    Find or create this user's PlatformCourse and the course's shared Module.
    `title`/`description` update the course metadata when given (course ingest);
    callers without course metadata (document upload) pass None to leave it unchanged.
    """
    # Always lock the course before the user (see _platform_user) — consistent order, no deadlock
    await _xact_lock(db, f"platform_course:{api_key.id}:{course_id}")
    platform_course = await _find_platform_course(db, api_key.id, course_id, user_id)

    module = None
    if platform_course and platform_course.module_id:
        module = await db.get(Module, platform_course.module_id)
    if module is None:
        module = await _shared_module(db, api_key.id, course_id)

    if module is None:
        owner  = await _platform_user(db, api_key.platform, user_id)
        module = Module(
            id=uuid.uuid4(),
            title=title or f"Course {course_id}",
            description=description,
            course_code=f"{api_key.platform[:3]}{course_id[:6]}".upper(),
            owner_id=owner.id,
            access_type="class",
            status="active",
            module_metadata={
                "source":             "platform",
                "platform":           api_key.platform,
                "platform_course_id": course_id,
            },
        )
        db.add(module)
        await db.flush()

    if title is not None:
        module.title       = title
        module.description = description

    if platform_course is None:
        platform_course = PlatformCourse(
            id=uuid.uuid4(),
            api_key_id=api_key.id,
            platform_course_id=course_id,
            platform_user_id=user_id,
            module_id=module.id,
            course_title=title or module.title,
            course_description=description,
            status="pending",
        )
        db.add(platform_course)
    else:
        platform_course.module_id = module.id
        if title is not None:
            platform_course.course_title       = title
            platform_course.course_description = description

    await db.commit()
    return platform_course, module


async def ingest_into_module(
    db:      AsyncSession,
    module:  Module,
    doc:     Document,
    content: bytes,
    version: int,
) -> dict:
    """
    Index `content` as `doc` (class material) in `module`: embeds into Typesense, replaces the
    doc's DocumentChunk rows and updates its status. Typesense chunk ids are
    `{doc.id}__{idx}`, so re-indexing a doc upserts in place; callers remove leftover chunks of
    the previous version afterwards. Caller commits.
    """
    result = await run_in_threadpool(
        get_ingestor().ingest,
        doc.filename,
        content,
        str(doc.id),
        module_id=str(module.id),
        course_code=module.course_code or "",
        visibility="class",
        doc_version=version,
        lecturer_id=str(module.owner_id),
    )

    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
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
    return result


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _latest_course_content(db: AsyncSession, module_id: uuid.UUID) -> Optional[tuple[ModuleDocument, Document]]:
    result = await db.execute(
        select(ModuleDocument, Document)
        .join(Document, Document.id == ModuleDocument.document_id)
        .where(
            ModuleDocument.module_id == module_id,
            Document.filename == COURSE_CONTENT_FILENAME,
            ModuleDocument.is_latest == True,  # noqa: E712
        )
        .order_by(ModuleDocument.created_at.desc())
        .limit(1)
    )
    row = result.first()
    return (row[0], row[1]) if row else None


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
            prev        = await _latest_course_content(db, module.id)
            new_version = prev[0].version + 1 if prev else 1

            content = course_text.encode("utf-8")
            doc = Document(
                id=uuid.uuid4(),
                owner_id=module.owner_id,
                filename=COURSE_CONTENT_FILENAME,
                file_type="txt",
                file_size_bytes=len(content),
                visibility="class",
                status="pending",
                doc_metadata={
                    "source":             "platform",
                    "platform_course_id": pc.platform_course_id,
                    "content_hash":       content_hash(course_text),
                },
            )
            db.add(doc)
            await db.commit()

            # Index the new version BEFORE retiring the old one — the module is shared, so
            # other users keep getting answers from the old content until the new one is live
            result = await ingest_into_module(db, module, doc, content, new_version)

            if result["status"] == "indexed":
                db.add(ModuleDocument(
                    id=uuid.uuid4(),
                    module_id=module.id,
                    document_id=doc.id,
                    uploaded_by=module.owner_id,
                    version=new_version,
                    is_latest=True,
                    visibility="class",
                ))
                if prev:
                    prev_md, prev_doc = prev
                    prev_md.is_latest = False
                    await run_in_threadpool(
                        delete_document_chunks, get_typesense_client(), str(prev_doc.id),
                    )
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


# ── Platform chat persistence ──────────────────────────────────────────────
# One ChatSession per (platform user, course module), owned by the synthetic platform
# User — the existing chat schema, no platform-specific columns needed.

async def _course_module_ids(
    db: AsyncSession, api_key_id: uuid.UUID, course_id: str, user_id: str,
) -> list[uuid.UUID]:
    """Every module this course's content may live in (normally just the shared one)."""
    ids: list[uuid.UUID] = []
    pc = await _find_platform_course(db, api_key_id, course_id, user_id)
    if pc and pc.module_id:
        ids.append(pc.module_id)
    shared = await _shared_module(db, api_key_id, course_id)
    if shared:
        ids.append(shared.id)
    doc_modules = await db.execute(
        select(PlatformDocument.module_id).where(
            PlatformDocument.api_key_id         == api_key_id,
            PlatformDocument.platform_course_id == course_id,
            PlatformDocument.module_id.is_not(None),
        ).distinct()
    )
    ids.extend(doc_modules.scalars().all())
    return list(dict.fromkeys(ids))


async def _latest_chat_session(
    db: AsyncSession, owner_id: uuid.UUID, module_ids: list[uuid.UUID],
) -> Optional[ChatSession]:
    if not module_ids:
        return None
    result = await db.execute(
        select(ChatSession)
        .where(
            ChatSession.user_id   == owner_id,
            ChatSession.module_id.in_(module_ids),
            ChatSession.is_active == True,  # noqa: E712
        )
        .order_by(ChatSession.updated_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def _get_or_create_chat_session(
    db: AsyncSession, identity: PlatformIdentity, course: ReadyCourse,
) -> ChatSession:
    owner = await _platform_user(db, identity.api_key.platform, course.platform_user_id)
    await _xact_lock(db, f"platform_chat:{owner.id}:{course.module_id}")
    session = await _latest_chat_session(db, owner.id, [course.module_id])
    if session is None:
        session = ChatSession(
            id=uuid.uuid4(),
            user_id=owner.id,
            module_id=course.module_id,
            title=f"{course.course_title}"[:255],
            session_metadata={
                "source":             "platform",
                "api_key_id":         str(identity.api_key.id),
                "platform_course_id": course.platform_course_id,
                "platform_user_id":   course.platform_user_id,
            },
        )
        db.add(session)
        await db.flush()
    return session


async def _recent_turns(db: AsyncSession, session_id: uuid.UUID, turns: int) -> list[dict]:
    """The last `turns` Q&A pairs, oldest first — conversation context for the model."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(turns * 2)
    )
    return [{"role": m.role, "content": m.content} for m in reversed(result.scalars().all())]


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

    # The module is shared by the whole course: if another user already indexed identical
    # content, this user is ready immediately — no re-embedding
    course_text = build_course_text(req)
    latest      = await _latest_course_content(db, module.id)
    if latest:
        _, latest_doc = latest
        same = (latest_doc.doc_metadata or {}).get("content_hash") == content_hash(course_text)
        if same and latest_doc.status == "indexed":
            platform_course.status        = "ready"
            platform_course.chunk_count   = latest_doc.chunk_count
            platform_course.indexed_at    = latest_doc.indexed_at or datetime.now(timezone.utc)
            platform_course.error_message = None
            await db.commit()
            return {
                "module_id": str(module.id),
                "course_id": course_id,
                "status":    "ready",
                "message":   "Course content already indexed.",
            }

    platform_course.status = "indexing"
    await db.commit()

    background_tasks.add_task(index_course_content, platform_course.id, course_text)

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
    """
    AI Tutor — answer a question scoped to a platform course. The conversation is stored
    per course + user (see /chat/history), and recent turns are sent as context.
    """
    course  = await _get_ready_course(db, identity, req.course_id, req.user_id)
    session = await _get_or_create_chat_session(db, identity, course)
    history = await _recent_turns(db, session.id, CHAT_HISTORY_TURNS)

    db.add(ChatMessage(session_id=session.id, role="user", content=req.message))
    await db.commit()  # save the question (and release locks) before the slow LLM call

    from app.agents.rag_pipeline import get_pipeline
    result = await run_in_threadpool(
        get_pipeline().query,
        question=req.message,
        history=history,
        module_id=str(course.module_id),
        scope_mode="everything",
        complexity=req.complexity,
        # Broad questions about the course/material should still get an answer from its content
        fallback_top_k=4,
    )

    db.add(ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        sources=[s.model_dump() for s in result["sources"]],
        latency_ms=result.get("latency_ms"),
        token_count=result.get("token_count"),
    ))
    session.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {
        "answer":           result["answer"],
        "sources":          result["sources"],
        "no_content_found": result["no_content_found"],
        "session_id":       str(session.id),
    }


@router.get("/chat/history")
async def get_chat_history(
    course_id: OptionalId = None,
    user_id:   OptionalId = None,
    limit:     int = Query(default=50, ge=1, le=200),
    db:        AsyncSession = Depends(get_db),
    identity:  PlatformIdentity = Depends(get_platform_identity),
):
    """
    The user's saved conversation for a course (latest `limit` messages, oldest first),
    so the panel can restore it after a reload or a new login. Empty if none yet.
    """
    course_id, user_id = identity.scope(course_id, user_id)

    owner = (await db.execute(
        select(User).where(User.email == platform_user_email(identity.api_key.platform, user_id))
    )).scalar_one_or_none()
    session = None
    if owner:
        module_ids = await _course_module_ids(db, identity.api_key.id, course_id, user_id)
        session = await _latest_chat_session(db, owner.id, module_ids)
    if session is None:
        return {"session_id": None, "messages": []}

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    messages = list(reversed(result.scalars().all()))
    return {
        "session_id": str(session.id),
        "messages": [{
            "id":        str(m.id),
            "role":      m.role,
            "content":   m.content,
            "sources":   m.sources or [],
            "timestamp": m.created_at.isoformat(),
        } for m in messages],
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
            complexity=req.complexity,
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
            complexity=req.complexity,
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
    """
    Generate a Markdown summary for a platform course. The latest summary per
    course + user + topic is saved (see GET /summary) and replaced on regeneration.
    """
    pc = await _get_ready_course(db, identity, req.course_id, req.user_id)

    from app.agents.ai_features import generate_summary
    try:
        summary, _chunk_count = await run_in_threadpool(
            generate_summary,
            module_id=str(pc.module_id),
            scope="module",
            topic=req.topic,
            complexity=req.complexity,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    saved = await _save_summary(db, identity, pc, (req.topic or "").strip(), req.complexity, summary)
    return {"course_id": pc.platform_course_id, **_summary_payload(saved)}


async def _save_summary(
    db: AsyncSession, identity: PlatformIdentity, course: ReadyCourse,
    topic: str, complexity: str, content: str,
) -> PlatformSummary:
    """Upsert the latest summary for course + user + topic ('' = all content)."""
    api_key_id = identity.api_key.id
    await _xact_lock(db, f"platform_summary:{api_key_id}:{course.platform_course_id}:{course.platform_user_id}:{topic}")
    result = await db.execute(
        select(PlatformSummary).where(
            PlatformSummary.api_key_id == api_key_id,
            PlatformSummary.course_id  == course.platform_course_id,
            PlatformSummary.user_id    == course.platform_user_id,
            PlatformSummary.topic      == topic,
        )
    )
    row = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if row is None:
        row = PlatformSummary(
            id=uuid.uuid4(), api_key_id=api_key_id, course_id=course.platform_course_id,
            user_id=course.platform_user_id, topic=topic,
        )
        db.add(row)
    row.content    = content
    row.complexity = complexity
    row.created_at = now
    await db.commit()
    return row


def _summary_payload(row: Optional[PlatformSummary]) -> dict:
    return {
        "summary":    row.content if row else None,
        "topic":      (row.topic or None) if row else None,   # '' → null = all content
        "complexity": row.complexity if row else None,
        "created_at": row.created_at.isoformat() if row and row.created_at else None,
    }


@router.get("/summary")
async def get_summary(
    course_id: OptionalId = None,
    user_id:   OptionalId = None,
    topic:     Optional[str] = Query(default=None, description="Omit for the most recent summary on any topic"),
    db:        AsyncSession = Depends(get_db),
    identity:  PlatformIdentity = Depends(get_platform_identity),
):
    """
    The user's saved summary for a course: for `topic` if given ('' = all content),
    otherwise the most recent one on any topic. All fields null if there's none.
    """
    course_id, user_id = identity.scope(course_id, user_id)
    query = select(PlatformSummary).where(
        PlatformSummary.api_key_id == identity.api_key.id,
        PlatformSummary.course_id  == course_id,
        PlatformSummary.user_id    == user_id,
    )
    if topic is not None:
        query = query.where(PlatformSummary.topic == topic.strip())
    result = await db.execute(query.order_by(PlatformSummary.created_at.desc()).limit(1))
    return _summary_payload(result.scalars().first())
