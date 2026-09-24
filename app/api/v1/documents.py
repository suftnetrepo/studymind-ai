"""
v1 Platform documents — tutors upload course materials (PDF/DOCX/TXT/MD) that are stored in
Cloudinary and indexed into the course's shared module, so every student's AI features use them.

Auth: API key (server-to-server, trusted) or session token. With a session token, upload /
replace / delete require user_role tutor or admin; listing is open to any course member.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.api.v1.platform import get_or_create_module, ingest_into_module
from app.auth.api_key_auth import PlatformIdentity, get_platform_identity
from app.config import get_settings
from app.db.engine import AsyncSessionLocal, get_db
from app.db.models import Document, Module, ModuleDocument, PlatformDocument
from app.logging_config import get_logger
from app.retrieval.typesense_client import (
    delete_document_chunks, get_typesense_client, mark_chunks_superseded,
)
from app.storage import cloudinary_client as storage

log = get_logger(__name__)

router = APIRouter(prefix="/v1/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE      = 20 * 1024 * 1024  # 20 MB
MANAGER_ROLES      = {"tutor", "admin"}


# ── Helpers ────────────────────────────────────────────────────────────────

def _course_scope(identity: PlatformIdentity, course_id: Optional[str]) -> str:
    """Course the request acts on — the token's course for session tokens."""
    if identity.auth_type == "session_token":
        return identity.scope(course_id, None)[0]
    if not course_id:
        raise HTTPException(status_code=422, detail="course_id is required")
    return course_id


def _require_manager(identity: PlatformIdentity) -> None:
    if identity.auth_type == "session_token" and identity.user_role not in MANAGER_ROLES:
        raise HTTPException(status_code=403, detail="Only tutors and admins can manage course materials")


def _require_storage() -> None:
    if not get_settings().cloudinary_configured:
        raise HTTPException(status_code=503, detail="Document storage is not configured")


async def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    """Validate extension and size; returns (filename, bytes)."""
    filename = Path(file.filename or "").name or "document"
    ext      = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not supported. Allowed: PDF, DOCX, TXT, MD")

    # Read at most MAX+1 bytes so oversized uploads aren't fully buffered
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    return filename, content


async def _get_document(
    db: AsyncSession, identity: PlatformIdentity, document_id: uuid.UUID, course_id: str,
) -> PlatformDocument:
    result = await db.execute(
        select(PlatformDocument).where(
            PlatformDocument.id                 == document_id,
            PlatformDocument.api_key_id         == identity.api_key.id,
            PlatformDocument.platform_course_id == course_id,
        )
    )
    pdoc = result.scalar_one_or_none()
    if not pdoc:
        raise HTTPException(status_code=404, detail="Document not found")
    return pdoc


def _serialize(d: PlatformDocument) -> dict:
    return {
        "id":          str(d.id),
        "filename":    d.filename,
        "status":      d.status,
        "chunk_count": d.chunk_count,
        "size":        d.file_size_bytes,
        "format":      d.file_format,
        "url":         d.cloudinary_url,
        "uploaded_by": d.uploaded_by,
        "indexed_at":  d.indexed_at.isoformat() if d.indexed_at else None,
        "error":       d.error_message,
        "created_at":  d.created_at.isoformat() if d.created_at else None,
    }


async def index_platform_document(platform_document_id: uuid.UUID, content: bytes) -> None:
    """
    Background task: index an uploaded (or replaced) file into its course module.
    Uses its own DB session. New chunks are upserted before the previous version's leftovers
    are removed, so a replaced document never disappears from search mid-index.
    """
    async with AsyncSessionLocal() as db:
        pdoc = await db.get(PlatformDocument, platform_document_id)
        if not pdoc:
            return
        try:
            doc    = await db.get(Document, pdoc.document_id) if pdoc.document_id else None
            module = await db.get(Module, pdoc.module_id) if pdoc.module_id else None
            if not doc or not module:
                raise RuntimeError("Document or module record missing")

            md_result = await db.execute(
                select(ModuleDocument).where(ModuleDocument.document_id == doc.id)
            )
            md          = md_result.scalars().first()
            old_version = md.version if md else None
            new_version = old_version + 1 if old_version else 1

            pdoc.status = "indexing"
            await db.commit()

            result = await ingest_into_module(db, module, doc, content, new_version)
            ts     = get_typesense_client()

            if old_version:
                # Remove the previous version's chunks — upserted indices were overwritten,
                # this clears the rest (or all of them if the new version failed to index)
                await run_in_threadpool(mark_chunks_superseded, ts, str(doc.id), old_version)

            if result["status"] == "indexed":
                if md:
                    md.version   = new_version
                    md.is_latest = True
                else:
                    db.add(ModuleDocument(
                        id=uuid.uuid4(),
                        module_id=module.id,
                        document_id=doc.id,
                        uploaded_by=module.owner_id,
                        version=new_version,
                        is_latest=True,
                        visibility="class",
                    ))
                pdoc.status        = "ready"
                pdoc.chunk_count   = result["chunk_count"]
                pdoc.indexed_at    = datetime.now(timezone.utc)
                pdoc.error_message = None
            else:
                pdoc.status        = "failed"
                pdoc.chunk_count   = 0
                pdoc.error_message = result["error"]
            await db.commit()
            log.info("platform_document_indexed", id=str(pdoc.id), status=pdoc.status,
                     chunks=result["chunk_count"], version=new_version)

        except Exception as exc:
            log.error("platform_document_index_failed", id=str(platform_document_id), error=str(exc))
            await db.rollback()
            pdoc = await db.get(PlatformDocument, platform_document_id)
            if pdoc:
                pdoc.status        = "failed"
                pdoc.error_message = str(exc)[:1000]
                await db.commit()


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/upload", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    course_id:        Optional[str] = Form(default=None),
    user_id:          Optional[str] = Form(default=None),
    db:               AsyncSession = Depends(get_db),
    identity:         PlatformIdentity = Depends(get_platform_identity),
):
    """Upload a course document: stored in Cloudinary, then indexed in the background."""
    _require_manager(identity)
    course_id, user_id = identity.scope(course_id, user_id)
    _require_storage()
    filename, content = await _read_upload(file)

    _, module = await get_or_create_module(db, identity.api_key, course_id, user_id)

    pdoc_id = uuid.uuid4()
    try:
        stored = await storage.upload_document(
            content, filename, identity.api_key.platform, course_id, str(pdoc_id),
        )
    except Exception as e:
        log.error("document_upload_storage_failed", error=str(e))
        raise HTTPException(status_code=502, detail="Failed to upload file to storage")

    ext = Path(filename).suffix.lower().lstrip(".")
    try:
        doc = Document(
            id=uuid.uuid4(),
            owner_id=module.owner_id,
            filename=filename,
            file_type=ext,
            file_size_bytes=len(content),
            visibility="class",
            status="pending",
            doc_metadata={"source": "platform_upload", "platform_course_id": course_id},
        )
        db.add(doc)
        pdoc = PlatformDocument(
            id=pdoc_id,
            api_key_id=identity.api_key.id,
            platform_course_id=course_id,
            uploaded_by=user_id,
            cloudinary_public_id=stored["public_id"],
            cloudinary_url=stored["secure_url"],
            filename=filename,
            file_size_bytes=stored["bytes"],
            file_format=stored["format"] or ext,
            module_id=module.id,
            document_id=doc.id,
            status="pending",
        )
        db.add(pdoc)
        await db.commit()
    except Exception:
        await storage.delete_document(stored["public_id"])  # don't orphan the stored file
        raise

    background_tasks.add_task(index_platform_document, pdoc.id, content)

    return {**_serialize(pdoc), "status": "indexing"}


@router.get("/list")
async def list_documents(
    course_id: Optional[str] = None,
    db:        AsyncSession = Depends(get_db),
    identity:  PlatformIdentity = Depends(get_platform_identity),
):
    """List a course's uploaded documents, newest first."""
    course_id = _course_scope(identity, course_id)
    result = await db.execute(
        select(PlatformDocument)
        .where(
            PlatformDocument.api_key_id         == identity.api_key.id,
            PlatformDocument.platform_course_id == course_id,
        )
        .order_by(PlatformDocument.created_at.desc())
    )
    return [_serialize(d) for d in result.scalars().all()]


@router.delete("/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    course_id:   Optional[str] = None,
    db:          AsyncSession = Depends(get_db),
    identity:    PlatformIdentity = Depends(get_platform_identity),
):
    """Delete a document: its Cloudinary file, search chunks and records."""
    _require_manager(identity)
    course_id = _course_scope(identity, course_id)
    pdoc      = await _get_document(db, identity, document_id, course_id)

    if pdoc.document_id:
        try:
            await run_in_threadpool(delete_document_chunks, get_typesense_client(), str(pdoc.document_id))
        except Exception as e:
            log.error("document_chunks_delete_failed", id=str(pdoc.id), error=str(e))
        # Core DELETE so Postgres' ON DELETE CASCADE removes DocumentChunk + ModuleDocument rows
        # (an ORM delete would lazy-load the chunk collection, which async sessions can't do)
        await db.execute(delete(Document).where(Document.id == pdoc.document_id))

    await storage.delete_document(pdoc.cloudinary_public_id)
    await db.delete(pdoc)
    await db.commit()
    return {"deleted": True, "document_id": str(document_id)}


@router.post("/{document_id}/replace", status_code=202)
async def replace_document(
    document_id:      uuid.UUID,
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    course_id:        Optional[str] = Form(default=None),
    db:               AsyncSession = Depends(get_db),
    identity:         PlatformIdentity = Depends(get_platform_identity),
):
    """Replace a document's file with a new version and re-index it."""
    _require_manager(identity)
    course_id = _course_scope(identity, course_id)
    pdoc      = await _get_document(db, identity, document_id, course_id)
    _require_storage()
    filename, content = await _read_upload(file)

    try:
        stored = await storage.upload_document(
            content, filename, identity.api_key.platform, course_id, str(pdoc.id),
        )
    except Exception as e:
        log.error("document_replace_storage_failed", error=str(e))
        raise HTTPException(status_code=502, detail="Failed to upload file to storage")

    # Same filename → same public_id, already overwritten; only delete a different old file
    if stored["public_id"] != pdoc.cloudinary_public_id:
        await storage.delete_document(pdoc.cloudinary_public_id)

    ext = Path(filename).suffix.lower().lstrip(".")
    pdoc.cloudinary_public_id = stored["public_id"]
    pdoc.cloudinary_url       = stored["secure_url"]
    pdoc.filename             = filename
    pdoc.file_size_bytes      = stored["bytes"]
    pdoc.file_format          = stored["format"] or ext
    pdoc.status               = "pending"
    pdoc.chunk_count          = 0
    pdoc.indexed_at           = None
    pdoc.error_message        = None

    doc = await db.get(Document, pdoc.document_id) if pdoc.document_id else None
    if doc:
        doc.filename        = filename
        doc.file_type       = ext
        doc.file_size_bytes = len(content)
        doc.status          = "pending"
    await db.commit()

    background_tasks.add_task(index_platform_document, pdoc.id, content)

    return {**_serialize(pdoc), "status": "indexing"}
