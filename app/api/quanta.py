"""
Quanta router — FastAPI endpoints for the Quanta STEM voice tutor.

Mounts at /api/quanta/...
Auth: Bearer JWT (same tokens as StudyMind — shared secret).

Endpoints:
  POST /api/quanta/curriculum/upload   — ingest a syllabus PDF for a subject+exam board
  GET  /api/quanta/curriculum          — list curricula for the current user
  POST /api/quanta/chat                — Socratic RAG Q&A (non-streaming)
  POST /api/quanta/voice-context       — fetch topic chunks for an OpenAI Realtime session
  GET  /api/quanta/topics              — list topics from an indexed curriculum
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.agents.llm_factory import get_embed_model, get_llm
from app.auth.dependencies import require_auth
from app.config import get_settings
from app.db.engine import get_db
from app.db.models import Document, User
from app.ingestion.ingestor import ALLOWED_EXTENSIONS, get_ingestor
from app.logging_config import get_logger
from app.retrieval.retriever import SearchScope, ScopedHybridRetriever
from llama_index.core.schema import QueryBundle

log      = get_logger(__name__)
router   = APIRouter(prefix="/api/quanta", tags=["quanta"])
executor = ThreadPoolExecutor(max_workers=2)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# ── Exam board / country enums ────────────────────────────────────────────────

EXAM_BOARDS = {
    "uk":    ["AQA", "Edexcel", "OCR", "WJEC", "Cambridge"],
    "ng":    ["WAEC", "NECO", "JAMB", "NABTEB"],
    "us":    ["AP", "Common Core", "SAT", "ACT"],
    "gh":    ["WASSCE", "BECE", "Cambridge"],
    "other": ["IB", "Other"],
}

SUBJECTS = ["maths", "physics", "chemistry"]
LEVELS   = ["gcse", "a-level", "ib", "university", "primary", "secondary", "other"]

# ── Socratic system prompt — distinct from StudyMind's "Revvo" ───────────────

SOCRATIC_SYSTEM = """\
You are Quanta — a voice-first STEM tutor that never gives answers directly.
You teach using the Socratic method: ask questions that guide the student to
discover the answer themselves.

Rules:
- NEVER give the complete answer outright. Break problems into small steps.
- Ask one focused question at a time.
- When a student is stuck, give a hint — not the answer.
- Celebrate partial progress ("Good — you've identified X. Now what about Y?")
- Stay strictly within the student's curriculum. Do not introduce topics outside
  the uploaded syllabus.
- Use precise mathematical / scientific language. Format equations in plain text
  (e.g. "integral of 2x dx" not LaTeX).
- Keep responses short — this is a voice interface. Aim for 2-4 sentences max.
- You teach: {subject} at {level} level, {exam_board} specification, {country}.
"""


def build_socratic_prompt(
    subject: str,
    level: str,
    exam_board: str,
    country: str,
    history: list[dict],
    context: str,
    question: str,
) -> str:
    system = SOCRATIC_SYSTEM.format(
        subject=subject, level=level, exam_board=exam_board, country=country
    )

    history_text = ""
    if history:
        lines = []
        for msg in history[-8:]:
            role = msg.get("role", "student").capitalize()
            lines.append(f"{role}: {msg.get('content', '')}")
        history_text = "\n".join(lines)

    return f"""{system}

--- Conversation so far ---
{history_text or "(new session)"}
---------------------------

--- Curriculum content (use ONLY this) ---
{context or "(no curriculum uploaded yet — gently tell the student to upload their syllabus)"}
------------------------------------------

Student: {question}

Quanta:"""


# ── Schemas ───────────────────────────────────────────────────────────────────

class CurriculumMeta(BaseModel):
    subject:    str = Field(..., description="maths | physics | chemistry")
    country:    str = Field(..., description="uk | ng | us | gh | other")
    exam_board: str = Field(..., description="e.g. AQA, WAEC, AP")
    level:      str = Field(..., description="gcse | a-level | ib | university ...")
    name:       str | None = None
    exam_year:  int | None = None


class ChatRequest(BaseModel):
    subject:    str
    level:      str
    exam_board: str
    country:    str
    question:   str
    history:    list[dict] = []
    module_id:  str | None = None   # document_id of the indexed curriculum
    top_k:      int = 6


class ChatResponse(BaseModel):
    answer:      str
    sources:     list[dict] = []
    latency_ms:  int


class VoiceContextRequest(BaseModel):
    topic:      str                 # e.g. "Integration by substitution"
    subject:    str
    level:      str
    exam_board: str
    country:    str
    module_id:  str | None = None
    top_k:      int = 8


class VoiceContextResponse(BaseModel):
    context:     str                # assembled text for the Realtime API system prompt
    chunk_count: int
    topic:       str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _retrieve_chunks(
    query: str,
    student_id: str,
    module_id: str | None,
    top_k: int,
) -> tuple[list, str]:
    """Retrieve top-k chunks and assemble context string."""
    embed_model = get_embed_model()
    scope = SearchScope(
        student_id=student_id,
        module_id=module_id,
        current_semester_only=False,
        latest_only=True,
        include_class=True,
        include_personal=True,
    )
    retriever = ScopedHybridRetriever(
        embed_model=embed_model,
        scope=scope,
        top_k=top_k,
        score_threshold=0.0,
    )
    nodes = retriever.retrieve(QueryBundle(query_str=query))
    parts = []
    for nws in nodes:
        meta  = nws.node.metadata
        fname = meta.get("filename", "doc")
        cidx  = meta.get("chunk_index", 0)
        parts.append(f"[{fname}, chunk {cidx}]\n{nws.node.text}")
    return nodes, "\n\n".join(parts)


def _ingest_curriculum(
    doc_id: str,
    filename: str,
    content: bytes,
    meta: CurriculumMeta,
    owner_id: str,
) -> dict:
    """Run in thread pool — synchronous ingestion."""
    settings = get_settings()
    engine   = create_engine(settings.postgres_dsn_sync)
    ingestor = get_ingestor()

    result = ingestor.ingest(
        filename=filename,
        content=content,
        document_id=doc_id,
        visibility="personal",
        owner_id=owner_id,
        is_current_semester=True,
    )

    with Session(engine) as session:
        doc = session.get(Document, uuid.UUID(doc_id))
        if doc:
            doc.status      = result["status"]
            doc.chunk_count = result.get("chunk_count", 0)
            doc.indexed_at  = result.get("indexed_at")
            # Stash Quanta metadata in doc_metadata JSONB
            doc.doc_metadata = {
                **(doc.doc_metadata or {}),
                "quanta": {
                    "subject":    meta.subject,
                    "country":    meta.country,
                    "exam_board": meta.exam_board,
                    "level":      meta.level,
                    "name":       meta.name,
                    "exam_year":  meta.exam_year,
                },
            }
            session.commit()
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/curriculum/upload", status_code=202)
async def upload_curriculum(
    file:     UploadFile            = File(...),
    subject:  str                   = Form(...),
    country:  str                   = Form(...),
    exam_board: str                 = Form(...),
    level:    str                   = Form(...),
    name:     str | None            = Form(None),
    exam_year: int | None           = Form(None),
    current_user: User              = Depends(require_auth),
    db: AsyncSession                = Depends(get_db),
):
    """
    Upload and index a syllabus PDF/DOCX.
    The file is chunked, embedded, and indexed into Typesense under the user's scope.
    Returns immediately with a document_id; indexing completes asynchronously.
    """
    # Validate
    if subject not in SUBJECTS:
        raise HTTPException(422, f"subject must be one of {SUBJECTS}")

    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(422, f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large — max 50 MB")

    meta = CurriculumMeta(
        subject=subject, country=country, exam_board=exam_board,
        level=level, name=name, exam_year=exam_year,
    )

    # Create pending document row
    doc_id  = str(uuid.uuid4())
    doc_row = Document(
        id=uuid.UUID(doc_id),
        owner_id=current_user.id,
        filename=file.filename or "syllabus",
        file_type=suffix.lstrip("."),
        file_size_bytes=len(content),
        visibility="personal",
        status="pending",
        doc_metadata={"quanta": meta.model_dump()},
    )
    db.add(doc_row)
    await db.commit()

    # Kick off ingestion in background thread
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        _ingest_curriculum,
        doc_id, file.filename or "syllabus", content, meta, str(current_user.id),
    )

    log.info("quanta_curriculum_upload", doc_id=doc_id, subject=subject, exam_board=exam_board)
    return {"document_id": doc_id, "status": "indexing", "message": "Indexing started — ready in ~30s"}


@router.get("/curriculum")
async def list_curricula(
    current_user: User       = Depends(require_auth),
    db: AsyncSession         = Depends(get_db),
):
    """List all indexed curricula for the current user."""
    result = await db.execute(
        select(Document)
        .where(Document.owner_id == current_user.id)
        .where(Document.doc_metadata["quanta"].isnot(None))
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return {
        "curricula": [
            {
                "id":          str(d.id),
                "filename":    d.filename,
                "status":      d.status,
                "chunk_count": d.chunk_count,
                "indexed_at":  d.indexed_at.isoformat() if d.indexed_at else None,
                "meta":        (d.doc_metadata or {}).get("quanta", {}),
                "created_at":  d.created_at.isoformat(),
            }
            for d in docs
        ]
    }


@router.post("/chat", response_model=ChatResponse)
async def socratic_chat(
    body:         ChatRequest,
    current_user: User         = Depends(require_auth),
):
    """
    Socratic RAG Q&A — returns a guiding question, not a direct answer.
    Used by the text-based chat fallback in Quanta.
    """
    import time
    t0 = time.perf_counter()

    nodes, context = _retrieve_chunks(
        query=body.question,
        student_id=str(current_user.id),
        module_id=body.module_id,
        top_k=body.top_k,
    )

    prompt = build_socratic_prompt(
        subject=body.subject,
        level=body.level,
        exam_board=body.exam_board,
        country=body.country,
        history=body.history,
        context=context,
        question=body.question,
    )

    llm      = get_llm()
    response = llm.complete(prompt)
    answer   = str(response).strip()

    sources = [
        {
            "filename":      nws.node.metadata.get("filename", "doc"),
            "chunk_index":   nws.node.metadata.get("chunk_index", 0),
            "snippet":       nws.node.text[:200],
            "relevance":     round(float(nws.score or 0), 4),
        }
        for nws in nodes[:3]
    ]

    latency = int((time.perf_counter() - t0) * 1000)
    log.info("quanta_chat", subject=body.subject, latency_ms=latency, chunks=len(nodes))

    return ChatResponse(answer=answer, sources=sources, latency_ms=latency)


@router.post("/voice-context", response_model=VoiceContextResponse)
async def voice_context(
    body:         VoiceContextRequest,
    current_user: User                 = Depends(require_auth),
):
    """
    Called before starting an OpenAI Realtime session.
    Returns assembled curriculum context for the given topic — ready to inject
    into the Realtime API system prompt.
    """
    _, context = _retrieve_chunks(
        query=body.topic,
        student_id=str(current_user.id),
        module_id=body.module_id,
        top_k=body.top_k,
    )

    # Build the full system prompt Quanta's voice client will use
    system_prompt = build_socratic_prompt(
        subject=body.subject,
        level=body.level,
        exam_board=body.exam_board,
        country=body.country,
        history=[],
        context=context,
        question=f"We are about to study: {body.topic}",
    )

    chunk_count = context.count("[") if context else 0
    log.info("quanta_voice_context", topic=body.topic, subject=body.subject, chunks=chunk_count)

    return VoiceContextResponse(
        context=system_prompt,
        chunk_count=chunk_count,
        topic=body.topic,
    )


class HintRequest(BaseModel):
    sessionId:   str
    quantaLine:  str
    lastStudent: str
    subject:     str
    topic:       str

class HintResponse(BaseModel):
    hint:     str
    concept:  str
    category: str  # scaffolding | correction | encouragement | deepening


@router.post("/hint", response_model=HintResponse)
async def generate_hint(
    body:         HintRequest,
    current_user: User         = Depends(require_auth),
):
    """
    Generate a live Socratic hint card after each Quanta response.
    Returns a brief coaching note for the student panel.
    """
    llm = get_llm()
    prompt = (
        f"You are an expert {body.subject} tutor. "
        f"The student is studying: {body.topic}.\n\n"
        f"Quanta just said: \"{body.quantaLine}\"\n"
        f"Student's last response: \"{body.lastStudent}\"\n\n"
        "Based on this exchange, generate a brief hint card for the student study panel.\n"
        "Return JSON with these fields ONLY:\n"
        "- hint: one helpful sentence (max 20 words) related to a key concept touched in the exchange\n"
        "- concept: 2-4 word concept name (e.g. 'Newton's Second Law')\n"
        "- category: one of 'scaffolding' | 'correction' | 'encouragement' | 'deepening'\n"
        "Respond with valid JSON only, no markdown."
    )
    import json as _json
    try:
        response = await llm.acomplete(prompt)
        raw      = str(response).strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = _json.loads(raw)
        return HintResponse(
            hint=data.get("hint", ""),
            concept=data.get("concept", body.topic),
            category=data.get("category", "scaffolding"),
        )
    except Exception:
        return HintResponse(hint="", concept=body.topic, category="scaffolding")


@router.get("/health")
async def health():
    """Quanta router health check."""
    return {"status": "ok", "service": "quanta"}
