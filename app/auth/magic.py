"""Single-use magic tokens for mobile → web handoff (quota top-up flow).

A mobile client that hits a quota limit calls POST /api/auth/magic-link
(JWT-authenticated) to get a short-lived raw token. It opens the web app
at /auth/magic?token=<raw>&redirect=/billing. The web calls
POST /api/auth/magic-verify (public) to exchange the token for a JWT pair,
sets its own session cookie, and redirects the user to /billing.

Tokens are valid for 10 minutes and can only be used once.
Raw tokens are never stored or logged.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone


def new_magic_token() -> tuple[str, str]:
    """Return (raw_token, token_hash). Store only the hash."""
    raw = secrets.token_urlsafe(48)
    return raw, _hash(raw)


def hash_magic_token(raw: str) -> str:
    return _hash(raw)


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def magic_token_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=10)
