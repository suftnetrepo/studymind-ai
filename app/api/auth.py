"""
Authentication endpoints — Sprint 1.
AUTH-01: Register
AUTH-02: Login → JWT access + refresh tokens
AUTH-03: Refresh access token
AUTH-04: Logout (revoke refresh token)
AUTH-08: Get current user profile (/me)
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_auth
from app.auth.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    refresh_token_expiry,
    verify_password,
)
from app.config import get_settings
from app.db.engine import get_db
from app.db.models import RefreshToken, User
from app.db.schemas import (
    ForgotPasswordRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserSchema,
    UserUpdateRequest,
)
from app.logging_config import get_logger

log    = get_logger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Public app config (which sign-up roles are open) ────────────────────────

@router.get("/config")
async def public_config():
    return {"enabled_roles": get_settings().enabled_role_list}


# ── AUTH-01: Register ──────────────────────────────────────────────────────

@router.post("/register", response_model=UserSchema, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user account. Role must be one of: lecturer | student | self_learner."""

    # Admin accounts are provisioned by staff (seed script / database), never self-served.
    if body.role == "admin":
        raise HTTPException(status_code=403, detail="Admin accounts cannot be created through registration")

    # Launch gating: only the roles switched on in ENABLED_ROLES can sign up.
    if body.role not in get_settings().enabled_role_list:
        raise HTTPException(
            status_code=403,
            detail="Sign-up for this account type isn't open yet. Choose Self-learner to get started.",
        )

    # Check email uniqueness
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    user = User(
        id=uuid.uuid4(),
        email=body.email,
        password_hash=hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
        is_active=True,
        is_verified=False,   # email verification — Phase 2
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    log.info("user_registered", user_id=str(user.id), role=user.role)
    return user


# ── Simple password reset (email + new password) ──────────────────────────
# Deliberately email-free for the launch stage. To limit abuse it only works for self-learner
# accounts (never staff or institution accounts), is rate limited, and signs out every device.

_RESET_ATTEMPTS: dict[str, list[float]] = {}


def _throttle(key: str, limit: int, window_s: int = 3600) -> bool:
    now = time.time()
    hits = [t for t in _RESET_ATTEMPTS.get(key, []) if now - t < window_s]
    if len(hits) >= limit:
        _RESET_ATTEMPTS[key] = hits
        return False
    hits.append(now)
    _RESET_ATTEMPTS[key] = hits
    return True


@router.post("/forgot-password")
async def forgot_password(
    body: ForgotPasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    ip = request.client.host if request.client else "unknown"
    email = body.email.lower()
    if not (_throttle(f"ip:{ip}", 20) and _throttle(f"email:{email}", 5)):
        raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")

    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="No account found with that email.")
    if user.role != "self_learner":
        raise HTTPException(
            status_code=403,
            detail="This account type can't reset its password in the app. Please contact support.",
        )

    user.password_hash = hash_password(body.new_password)
    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(RefreshToken).where(RefreshToken.user_id == user.id))
    await db.commit()
    log.info("password_reset_simple", user_id=str(user.id))
    return {"ok": True}


# ── AUTH-02: Login ─────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate and receive access + refresh tokens."""

    result = await db.execute(select(User).where(User.email == body.email))
    user   = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated — contact support",
        )

    # Issue tokens
    access_token          = create_access_token(str(user.id), user.role, user.email)
    raw_refresh, hashed   = generate_refresh_token()
    s                     = get_settings()

    rt = RefreshToken(
        user_id=user.id,
        token_hash=hashed,
        expires_at=refresh_token_expiry(),
        device_info=request.headers.get("User-Agent", "")[:255],
    )
    db.add(rt)

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    log.info("user_login", user_id=str(user.id), role=user.role)

    return TokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        token_type="bearer",
        expires_in=s.jwt_access_token_expire_minutes * 60,
    )


# ── AUTH-02: Refresh access token ─────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Exchange a valid refresh token for a new access token."""

    hashed = hash_refresh_token(body.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == hashed,
            RefreshToken.revoked == False,
        )
    )
    rt = result.scalar_one_or_none()

    invalid_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token",
    )

    if not rt:
        raise invalid_exc
    if rt.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        rt.revoked = True
        await db.commit()
        raise invalid_exc

    # Load user
    user_result = await db.execute(select(User).where(User.id == rt.user_id))
    user        = user_result.scalar_one_or_none()
    if not user or not user.is_active:
        raise invalid_exc

    # Rotate: revoke old, issue new
    rt.revoked           = True
    new_raw, new_hashed  = generate_refresh_token()
    s                    = get_settings()

    new_rt = RefreshToken(
        user_id=rt.user_id,
        token_hash=new_hashed,
        expires_at=refresh_token_expiry(),
        device_info=rt.device_info,
    )
    db.add(new_rt)

    new_access = create_access_token(str(user.id), user.role, user.email)
    await db.commit()

    log.info("token_refreshed", user_id=str(user.id))

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_raw,
        token_type="bearer",
        expires_in=s.jwt_access_token_expire_minutes * 60,
    )


# ── AUTH-04: Logout ────────────────────────────────────────────────────────

@router.post("/logout", status_code=204)
async def logout(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Revoke the provided refresh token. Client should discard the access token."""

    hashed = hash_refresh_token(body.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == hashed)
    )
    rt = result.scalar_one_or_none()
    if rt:
        rt.revoked = True
        await db.commit()
    # Always return 204 — do not reveal whether the token existed


# ── GET /me ────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserSchema)
async def get_me(current_user: User = Depends(require_auth)):
    """Return the currently authenticated user's profile."""
    return current_user


@router.patch("/me", response_model=UserSchema)
async def update_me(
    body: UserUpdateRequest,
    current_user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Update the current user's name or profile data."""
    if body.full_name is not None:
        current_user.full_name = body.full_name
    if body.profile is not None:
        current_user.profile = body.profile
    await db.commit()
    await db.refresh(current_user)
    return current_user


# ── DELETE /me (App Store guideline 5.1.1(v): in-app account deletion) ──────

@router.delete("/me", status_code=204)
async def delete_me(
    current_user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """
    Permanently delete the signed-in account and everything it owns: modules, documents, indexed
    chunks, chats, quizzes, flashcards, summaries and activity. Admins are removed by staff only.
    """
    if current_user.role == "admin":
        raise HTTPException(status_code=403, detail="Administrator accounts must be removed by staff")

    from sqlalchemy import delete as sa_delete, union
    from app.db.models import Document, Module, ModuleDocument
    from app.retrieval.typesense_client import delete_document_chunks, get_typesense_client

    owned_modules = select(Module.id).where(Module.owner_id == current_user.id)
    rows = await db.execute(union(
        select(Document.id).where(Document.owner_id == current_user.id),
        select(ModuleDocument.document_id).where(ModuleDocument.module_id.in_(owned_modules)),
    ))
    doc_ids = [str(r[0]) for r in rows.all()]

    # Remove the search index entries first; a failure here must not leave the account half-deleted.
    def _purge():
        client = get_typesense_client()
        for doc_id in doc_ids:
            try:
                delete_document_chunks(client, doc_id)
            except Exception as exc:
                log.warning("account_delete_chunk_purge_failed", document_id=doc_id, error=str(exc))

    import asyncio
    await asyncio.get_running_loop().run_in_executor(None, _purge)

    if doc_ids:
        await db.execute(sa_delete(Document).where(Document.id.in_([uuid.UUID(d) for d in doc_ids])))
    await db.delete(current_user)
    await db.commit()
    log.info("account_deleted", user_id=str(current_user.id), documents=len(doc_ids))
