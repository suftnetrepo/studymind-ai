"""Health check and system stats endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_db
from app.db.models import ChatMessage, ChatSession, Document, DocumentChunk
from app.db.schemas import HealthResponse, StatsResponse
from app.retrieval.typesense_client import get_typesense_client

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)):
    services: dict[str, bool] = {}

    # PostgreSQL
    try:
        await db.execute(select(func.now()))
        services["postgresql"] = True
    except Exception:
        services["postgresql"] = False

    # Typesense — v2 client uses collections list instead of health endpoint
    try:
        ts = get_typesense_client()
        ts.collections.retrieve()
        services["typesense"] = True
    except Exception:
        services["typesense"] = False

    all_ok = all(services.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        services=services,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats(db: AsyncSession = Depends(get_db)):
    total_sessions = (await db.execute(select(func.count()).select_from(ChatSession))).scalar() or 0
    total_messages = (await db.execute(select(func.count()).select_from(ChatMessage))).scalar() or 0
    total_docs     = (await db.execute(select(func.count()).select_from(Document))).scalar() or 0
    total_chunks   = (await db.execute(select(func.count()).select_from(DocumentChunk))).scalar() or 0

    avg_result = await db.execute(
        select(func.avg(ChatMessage.latency_ms))
        .where(ChatMessage.role == "assistant")
        .where(ChatMessage.latency_ms.is_not(None))
    )
    avg_latency = avg_result.scalar()

    return StatsResponse(
        total_sessions=total_sessions,
        total_messages=total_messages,
        total_documents=total_docs,
        total_chunks=total_chunks,
        avg_latency_ms=round(float(avg_latency), 1) if avg_latency else None,
    )
