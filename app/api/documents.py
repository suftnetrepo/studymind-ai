"""
Document management endpoints.
Ingestion runs in a thread pool using a synchronous SQLAlchemy session.
The ingestor returns pure data — no ORM objects cross thread boundaries.
"""
from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_db
from app.db.models import Document, DocumentChunk
from app.db.schemas import DocumentSchema, DocumentUploadResponse
from app.ingestion.ingestor import ALLOWED_EXTENSIONS, get_ingestor
from app.logging_config import get_logger
from app.retrieval.typesense_client import delete_document_chunks, get_typesense_client

log      = get_logger(__name__)
router   = APIRouter(prefix="/api/documents", tags=["documents"])
executor = ThreadPoolExecutor(max_workers=4)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def _ingest_and_persist(doc_id: str, filename: str, content: bytes) -> dict:
    """
    Runs in a thread pool with its own synchronous DB session.
    No async ORM objects — everything is plain data.
    """
    from app.config import get_settings
    settings = get_settings()
    engine   = create_engine(settings.postgres_dsn_sync)

    try:
        # Run ingestion — returns pure data dict, no ORM objects
        ingestor = get_ingestor()
        result   = ingestor.ingest(filename, content, doc_id)

        with Session(engine) as session:
            doc = session.get(Document, uuid.UUID(doc_id))
            if not doc:
                return {"status": "failed", "error": "Document record not found"}

            # Persist chunks
            for chunk in result["chunks"]:
                session.add(DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=uuid.UUID(doc_id),
                    typesense_id=chunk["typesense_id"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    token_count=chunk["token_count"],
                    chunk_metadata=chunk["chunk_metadata"],
                ))

            # Update document record
            doc.status        = result["status"]
            doc.chunk_count   = result["chunk_count"]
            doc.error_message = result["error"]
            doc.indexed_at    = result["indexed_at"]
            session.commit()

        return result

    except Exception as exc:
        log.error("ingest_persist_error", error=str(exc))
        try:
            with Session(engine) as session:
                doc = session.get(Document, uuid.UUID(doc_id))
                if doc:
                    doc.status        = "failed"
                    doc.error_message = str(exc)
                    session.commit()
        except Exception:
            pass
        return {"status": "failed", "chunk_count": 0, "error": str(exc)}
    finally:
        engine.dispose()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db:   AsyncSession = Depends(get_db),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    # Create document record asynchronously
    doc = Document(
        id=uuid.uuid4(),
        filename=file.filename,
        file_type=ext.lstrip("."),
        file_size_bytes=len(content),
        status="pending",
    )
    db.add(doc)
    await db.commit()
    doc_id = str(doc.id)

    # Run ingestion in thread pool (sync, no async ORM)
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, _ingest_and_persist, doc_id, file.filename, content
    )

    status      = result["status"]
    chunk_count = result.get("chunk_count", 0)
    error       = result.get("error") or ""

    log.info("upload_complete", filename=file.filename, status=status, chunks=chunk_count)

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=file.filename,
        status=status,
        message=(
            f"Indexed {chunk_count} chunks from '{file.filename}'"
            if status == "indexed"
            else f"Ingestion failed: {error}"
        ),
    )


@router.get("", response_model=list[DocumentSchema])
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return result.scalars().all()


@router.get("/{document_id}", response_model=DocumentSchema)
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc    = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc    = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    ts_client = get_typesense_client()
    delete_document_chunks(ts_client, str(document_id))
    await db.delete(doc)
    await db.commit()
    log.info("document_deleted", document_id=str(document_id))
