"""
FastAPI dependency functions for authentication and authorisation.
Use these as Depends() arguments on any protected endpoint.

Usage:
    @router.get("/protected")
    async def endpoint(current_user: User = Depends(require_auth)):
        ...

    @router.get("/lecturer-only")
    async def endpoint(current_user: User = Depends(require_lecturer)):
        ...
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import decode_access_token
from app.db.engine import get_db
from app.db.models import User

bearer_scheme = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Validates the Bearer JWT token and returns the authenticated User.
    Raises 401 if the token is missing, expired, or invalid.
    Raises 403 if the user account is inactive.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(credentials.credentials)
        user_id: str = payload.get("sub")
        if not user_id:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_id))
    user   = result.scalar_one_or_none()

    if not user:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )
    return user


# ── Role-based dependency factories ───────────────────────────────────────

def _require_role(*roles: str):
    """Returns a dependency that enforces one of the given roles."""
    async def dep(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {' or '.join(roles)}. "
                       f"Your role: {current_user.role}",
            )
        return current_user
    return dep


# Pre-built role dependencies — import and use directly
require_auth     = get_current_user                          # any authenticated user
require_admin    = _require_role("admin")                    # AUTH-07
require_lecturer = _require_role("lecturer", "admin")        # AUTH-06
require_student  = _require_role("student", "self_learner", "admin")  # AUTH-05
require_any      = get_current_user                          # alias for clarity
