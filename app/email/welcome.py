"""Welcome email sent once after registration."""
from __future__ import annotations

import structlog

from app.config import get_settings
from app.email.sender import email_configured, send_email
from app.email.templates import welcome_email

log = structlog.get_logger()


async def send_welcome_email(to_email: str, full_name: str, role: str) -> bool:
    s = get_settings()
    if not s.email_welcome_enabled or not email_configured():
        return False
    subject, html, text = welcome_email(full_name, role, s.email_app_url)
    ok = await send_email(to_email, full_name, subject, html, text)
    log.info("welcome_email", sent=ok, role=role)
    return ok
