"""Verified single-use password recovery. Raw tokens are never stored or logged."""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from app.config import get_settings
from app.email.sender import email_configured, send_email
from app.email.templates import reset_email


def new_recovery_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(48)
    return raw, hash_recovery_token(raw)


def hash_recovery_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def recovery_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=30)


def recovery_configured() -> bool:
    s = get_settings()
    return bool(s.password_recovery_enabled and email_configured() and s.recovery_public_url.startswith("https://"))


async def send_recovery_email(to_email: str, full_name: str, token: str) -> bool:
    """Never prints the URL, token, address or provider error. Returns whether the provider accepted the message."""
    s = get_settings()
    if not recovery_configured():
        raise RuntimeError("Password recovery delivery is not configured")
    url = s.recovery_public_url.rstrip("/") + "?token=" + quote(token, safe="")
    subject, html, text = reset_email(full_name, url)
    return await send_email(to_email, full_name, subject, html, text)
