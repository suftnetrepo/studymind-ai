"""
FastAPI application — StudyMind AI.
Sprint 1: Auth & JWT
Sprint 2: Institution & User Management
Sprint 3: Modules, Semesters, Weeks, Document Upload
Sprint 4: Scoped RAG Q&A with Citations
"""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.llm_factory import configure_llama_settings
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.features import router as features_router
from app.api.institutions import router as inst_router
from app.api.modules import router as modules_router
from app.api.writing import router as writing_router
from app.api.onboarding import router as onboarding_router
from app.api.activity import router as activity_router
from app.db.engine import create_all_tables
from app.logging_config import configure_logging, get_logger
from app.retrieval.typesense_client import ensure_collection, get_typesense_client

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("startup_begin", version="1.0.0", sprint="5")

    await create_all_tables()
    log.info("postgres_tables_ready")

    try:
        ts = get_typesense_client()
        ensure_collection(ts)
        log.info("typesense_collection_ready")
    except Exception as e:
        log.warning("typesense_unavailable", error=str(e))

    configure_llama_settings()
    log.info("llm_ready")
    log.info("startup_complete")
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="StudyMind AI",
        description=(
            "AI-powered study assistant — grounded answers from your own course materials.\n\n"
            "## Authentication\n"
            "All protected endpoints require: `Authorization: Bearer <access_token>`\n\n"
            "## Sprints complete\n"
            "- **Sprint 1** ✅ Auth & JWT\n"
            "- **Sprint 2** ✅ Institution & User Management\n"
            "- **Sprint 3** ✅ Modules, Semesters & Document Upload\n"
            "- **Sprint 4** ✅ Scoped RAG Q&A with Citations\n"
            "- **Sprint 5** — Quiz, Flashcards & Summariser (next)\n"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(inst_router)
    app.include_router(modules_router)
    app.include_router(features_router)
    app.include_router(chat_router)
    app.include_router(writing_router)
    app.include_router(onboarding_router)
    app.include_router(activity_router)
    # NOTE: the legacy unauthenticated /api/documents router is intentionally not mounted; all
    # document access goes through the authenticated /api/modules/{id}/documents endpoints.
    app.include_router(health_router)

    return app


app = create_app()
