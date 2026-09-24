"""
Chat API — Sprint 4: Scoped Q&A.

QA-01: Natural language Q&A grounded in uploaded materials
QA-02: Every answer includes source citations
QA-03: Word-by-word typewriter (SSE stream)
QA-04: 10-turn conversation memory
QA-05: Sessions persisted to PostgreSQL
QA-06: Sessions scoped per module
QA-07: No content found — clear message + suggestions
QA-08: Source citation tap-through (content_snippet)
SRCH-01–06: Scope control via scope_mode + module_id + shortcuts
SRCH-05: Scope indicator returned on every response
"""
from __future__ import annotations

import base64
import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.activity.tracker import log_activity
from app.agents.rag_pipeline import COMPLEXITY_MODIFIERS, get_pipeline
from app.auth.dependencies import require_auth
from app.config import get_settings
from app.db.engine import get_db
from app.db.models import ChatMessage, ChatSession, Module, Semester, User
from app.db.schemas import (
    MessageSchema, ScopedChatRequest, ScopedChatResponse,
    SessionCreate, SessionSchema, SessionWithMessages, SessionWithScope,
)
from app.logging_config import get_logger

log    = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


# ── Session helpers ────────────────────────────────────────────────────────

async def _get_or_create_session(
    session_id: uuid.UUID | None,
    module_id:  uuid.UUID | None,
    user: User,
    db: AsyncSession,
) -> ChatSession:
    if session_id:
        result  = await db.execute(
            select(ChatSession).where(
                ChatSession.id      == session_id,
                ChatSession.user_id == user.id,
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    # QA-06: new session scoped to module
    session = ChatSession(user_id=user.id, module_id=module_id)
    db.add(session)
    await db.flush()
    return session


async def _load_history(session_id: uuid.UUID, db: AsyncSession) -> list[dict]:
    """QA-04: Load last 10 turns as conversation history."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())  # newest 20, then back to chronological order
        .limit(20)
    )
    return [{"role": m.role, "content": m.content} for m in reversed(result.scalars().all())]


async def _resolve_module_scope(
    module_id: uuid.UUID | None,
    db: AsyncSession,
) -> tuple[str | None, str | None, str | None]:
    """
    Resolve module → (module_id_str, course_code, semester_label).
    Returns (None, None, None) if no module specified.
    """
    if not module_id:
        return None, None, None

    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()
    if not module:
        return str(module_id), None, None

    semester_label = None
    if module.semester_id:
        sem_result = await db.execute(
            select(Semester).where(Semester.id == module.semester_id)
        )
        sem = sem_result.scalar_one_or_none()
        if sem:
            semester_label = sem.label

    return str(module_id), module.course_code, semester_label


# ── QA-01: POST /api/chat ──────────────────────────────────────────────────

@router.post("/chat", response_model=ScopedChatResponse)
async def chat(
    req: ScopedChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    QA-01: Ask a question. Returns grounded answer with citations and scope indicator.
    Supports /csc109/week3 shortcuts in the message prefix.
    """
    session = await _get_or_create_session(
        req.session_id, req.module_id, current_user, db
    )
    history = await _load_history(session.id, db)

    module_id_str, course_code, semester_label = await _resolve_module_scope(
        req.module_id, db
    )

    settings = get_settings()
    pipeline = get_pipeline()

    # Save user message
    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=req.message,
    )
    db.add(user_msg)
    await db.flush()

    # Run scoped RAG query
    result = pipeline.query(
        question=req.message,
        history=history,
        top_k=req.top_k or settings.top_k_retrieval,
        student_id=str(current_user.id),
        module_id=module_id_str,
        course_code=course_code,
        scope_mode=req.scope_mode,
        include_archived=req.include_archived,
        semester_label=semester_label,
        complexity=req.complexity,
    )

    # Save assistant message with full metadata
    bot_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        sources=[s.model_dump() for s in result["sources"]],
        latency_ms=result["latency_ms"],
        token_count=result["token_count"],
    )
    db.add(bot_msg)

    # Auto-title session from first message
    if session.title == "New conversation":
        session.title = req.message[:60] + ("…" if len(req.message) > 60 else "")

    await db.commit()
    await db.refresh(bot_msg)
    await log_activity(current_user.id, "chat", req.module_id)

    return ScopedChatResponse(
        message_id=bot_msg.id,
        session_id=session.id,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
        token_count=result["token_count"],
        scope=result["scope"],
        no_content_found=result["no_content_found"],
        suggestions=result["suggestions"],
    )


# ── QA-03: POST /api/chat/stream ──────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(
    req: ScopedChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    QA-03: Streaming chat via SSE.
    Events: session → token (×N) → done (with sources + scope)
    """
    session = await _get_or_create_session(
        req.session_id, req.module_id, current_user, db
    )
    history = await _load_history(session.id, db)

    module_id_str, course_code, semester_label = await _resolve_module_scope(
        req.module_id, db
    )

    pipeline   = get_pipeline()
    settings   = get_settings()
    session_id = str(session.id)

    user_msg = ChatMessage(session_id=session.id, role="user", content=req.message)
    db.add(user_msg)
    await db.flush()
    await db.commit()
    await log_activity(current_user.id, "chat", req.module_id)

    async def event_stream():
        # Event 1: session ID so client can track
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        # Event 2: scope indicator so UI can show it before answer arrives
        t0          = time.perf_counter()
        full_answer = []

        try:
            stream, citations, scope_indicator = pipeline.stream_query(
                question=req.message,
                history=history,
                top_k=req.top_k or settings.top_k_retrieval,
                student_id=str(current_user.id),
                module_id=module_id_str,
                course_code=course_code,
                scope_mode=req.scope_mode,
                include_archived=req.include_archived,
                semester_label=semester_label,
                complexity=req.complexity,
            )

            # Send scope before first token (SRCH-05)
            yield f"data: {json.dumps({'type': 'scope', 'scope': scope_indicator.model_dump()})}\n\n"

            for chunk in stream:
                token = chunk.delta or ""
                if token:
                    full_answer.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        except Exception as exc:
            log.error("stream_error", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        latency_ms  = int((time.perf_counter() - t0) * 1000)
        answer_text = "".join(full_answer)
        sources_raw = [s.model_dump() for s in citations]

        no_content  = len(citations) == 0
        suggestions = []
        if no_content:
            from app.agents.rag_pipeline import _build_scope_from_request, _no_content_suggestions
            scope = _build_scope_from_request(
                student_id=str(current_user.id),
                module_id=module_id_str,
                course_code=course_code,
                scope_mode=req.scope_mode,
                include_archived=req.include_archived,
            )
            suggestions = _no_content_suggestions(scope)

        # Event N: done — full metadata
        yield f"data: {json.dumps({'type': 'done', 'sources': sources_raw, 'latency_ms': latency_ms, 'no_content_found': no_content, 'suggestions': suggestions})}\n\n"

        # Persist
        try:
            bot_msg = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=answer_text,
                sources=sources_raw,
                latency_ms=latency_ms,
            )
            db.add(bot_msg)
            if session.title == "New conversation":
                session.title = req.message[:60] + ("…" if len(req.message) > 60 else "")
            await db.commit()
        except Exception as exc:
            log.warning("stream_persist_failed", error=str(exc))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Session management ─────────────────────────────────────────────────────

@router.get("/sessions", response_model=list[SessionWithScope])
async def list_sessions(
    module_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    List the current user's sessions.
    QA-06: Optionally filter by module_id for module-scoped history.
    """
    query = select(ChatSession).where(
        ChatSession.user_id   == current_user.id,
        ChatSession.is_active == True,
    )
    if module_id:
        query = query.where(ChatSession.module_id == module_id)

    result   = await db.execute(query.order_by(ChatSession.updated_at.desc()).limit(50))
    sessions = result.scalars().all()

    out = []
    for s in sessions:
        count = (await db.execute(
            select(func.count()).where(ChatMessage.session_id == s.id)
        )).scalar() or 0
        schema               = SessionWithScope.model_validate(s)
        schema.message_count = count
        out.append(schema)
    return out


@router.post("/sessions", response_model=SessionWithScope, status_code=201)
async def create_session(
    body: SessionCreate,
    module_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """Create a new session, optionally scoped to a module."""
    session = ChatSession(
        title=body.title,
        user_id=current_user.id,
        module_id=module_id,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    schema               = SessionWithScope.model_validate(session)
    schema.message_count = 0
    return schema


@router.get("/sessions/{session_id}", response_model=SessionWithMessages)
async def get_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """QA-05: Return session with full message history."""
    result = await db.execute(
        select(ChatSession)
        .options(selectinload(ChatSession.messages))
        .where(
            ChatSession.id      == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id      == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)   # messages cascade; a deleted chat is really gone
    await db.commit()


# ── Scan & Solve / Voice input ─────────────────────────────────────────────

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_MAX_IMAGE_B64 = 14_000_000   # ~10 MB of image data
_MAX_AUDIO_B64 = 34_000_000   # ~25 MB (Whisper's own upload limit)


class ExtractImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=1)
    mime_type:    str = "image/jpeg"


class TranscribeRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)


def _openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


@router.post("/chat/extract-image")
async def extract_text_from_image(
    body: ExtractImageRequest,
    current_user: User = Depends(require_auth),
):
    """Scan & Solve: OCR a photo of a question/problem into editable text."""
    if body.mime_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type")
    if len(body.image_base64) > _MAX_IMAGE_B64:
        raise HTTPException(status_code=413, detail="Image is too large")

    try:
        resp = await _openai_client().chat.completions.create(
            model=get_settings().openai_chat_model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:{body.mime_type};base64,{body.image_base64}",
                    }},
                    {"type": "text", "text": (
                        "Extract all the text from this image exactly as it appears. "
                        "If it is a question or problem, extract it completely. "
                        "If it contains diagrams or equations, describe them clearly in text. "
                        "Return only the extracted content, nothing else."
                    )},
                ],
            }],
        )
        return {"text": (resp.choices[0].message.content or "").strip()}
    except Exception as exc:
        log.error("extract_image_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Could not read the image")


@router.post("/chat/transcribe")
async def transcribe_audio(
    body: TranscribeRequest,
    current_user: User = Depends(require_auth),
):
    """Voice input: transcribe a recorded clip with Whisper."""
    if len(body.audio_base64) > _MAX_AUDIO_B64:
        raise HTTPException(status_code=413, detail="Recording is too long")
    try:
        audio_bytes = base64.b64decode(body.audio_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio data")

    try:
        transcript = await _openai_client().audio.transcriptions.create(
            model="whisper-1",
            file=("recording.m4a", audio_bytes),
            language="en",
        )
        return {"text": transcript.text}
    except Exception as exc:
        log.error("transcribe_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Could not transcribe the recording")


# ── General chat: no module, no RAG ────────────────────────────────────────

class GeneralChatRequest(BaseModel):
    message:    str = Field(..., min_length=1, max_length=8192)
    session_id: uuid.UUID | None = None
    complexity: str = Field(default="normal", pattern="^(simple|normal|expert)$")


GENERAL_SYSTEM_PROMPT = """You are Revvo, a helpful and friendly AI assistant for students.
You can answer questions on any topic, academic or general.
You are NOT limited to course materials for this conversation.
{complexity}
Guidelines:
- Be helpful, accurate and concise
- If asked about mathematics, show working step by step
- If asked to write code, format it clearly in code blocks
- If asked for opinions, be balanced and educational
- Never make up facts; say you are unsure if you do not know
- Keep responses focused and well structured, using Markdown
"""


@router.post("/chat/general")
async def general_chat(
    req: GeneralChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """General-knowledge chat. Sessions with module_id=NULL are general chats."""
    session = await _get_or_create_session(req.session_id, None, current_user, db)
    history = await _load_history(session.id, db)

    system = GENERAL_SYSTEM_PROMPT.format(
        complexity=COMPLEXITY_MODIFIERS.get(req.complexity, COMPLEXITY_MODIFIERS["normal"])
    )
    started = time.monotonic()
    try:
        resp = await _openai_client().chat.completions.create(
            model=get_settings().openai_chat_model,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system},
                *history,
                {"role": "user", "content": req.message},
            ],
        )
    except Exception as e:
        log.error("general_chat_failed", error=str(e))
        raise HTTPException(status_code=502, detail="The AI service is unavailable. Try again.")

    answer = (resp.choices[0].message.content or "").strip()
    latency_ms = int((time.monotonic() - started) * 1000)

    db.add(ChatMessage(session_id=session.id, role="user", content=req.message))
    bot_msg = ChatMessage(
        session_id=session.id, role="assistant", content=answer,
        sources=[], latency_ms=latency_ms,
        token_count=resp.usage.total_tokens if resp.usage else None,
    )
    db.add(bot_msg)
    if session.title == "New conversation":
        session.title = req.message[:60] + ("…" if len(req.message) > 60 else "")
    await db.commit()
    await db.refresh(bot_msg)
    await log_activity(current_user.id, "chat", None)

    return {
        "answer":     answer,
        "session_id": str(session.id),
        "message_id": str(bot_msg.id),
        "sources":    [],
        "latency_ms": latency_ms,
    }
