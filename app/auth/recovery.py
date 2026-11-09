"""Verified single-use password recovery. Raw tokens are never stored or logged."""
from __future__ import annotations

import hashlib
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from urllib.parse import quote

from app.config import get_settings


def new_recovery_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(48)
    return raw, hash_recovery_token(raw)


def hash_recovery_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def recovery_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=30)


def recovery_configured() -> bool:
    s = get_settings()
    return bool(s.password_recovery_enabled and s.recovery_smtp_host
                and s.recovery_smtp_username and s.recovery_smtp_password
                and s.recovery_from_email and s.recovery_public_url.startswith("https://"))


def send_recovery_email(to_email: str, token: str) -> None:
    """Run via asyncio.to_thread; never print URL, token, or provider exception."""
    s = get_settings()
    if not recovery_configured():
        raise RuntimeError("Password recovery delivery is not configured")
    url = s.recovery_public_url.rstrip("/") + "?token=" + quote(token, safe="")
    message = EmailMessage()
    message["Subject"] = "Reset your StudyMind password"
    message["From"] = s.recovery_from_email
    message["To"] = to_email
    message.set_content("Use this link to reset your StudyMind password within 30 minutes:\\n\\n" + url
                        + "\\n\\nIf you did not request this, ignore this message.")
    with smtplib.SMTP(s.recovery_smtp_host, s.recovery_smtp_port, timeout=10) as smtp:
        smtp.starttls()
        smtp.login(s.recovery_smtp_username, s.recovery_smtp_password)
        smtp.send_message(message)
