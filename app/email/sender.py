"""Send one transactional email. Never logs addresses, content, links, keys or provider error text."""
from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage
from typing import Optional

import httpx
import structlog

from app.config import get_settings

log = structlog.get_logger()

BREVO_ENDPOINT = "https://api.brevo.com/v3/smtp/email"
_RETRIES = 3


def _from_address() -> str:
    s = get_settings()
    return (s.email_from_address or s.recovery_from_email or "").strip()


def email_provider() -> Optional[str]:
    """'brevo' when an API key and sender are set; 'smtp' when the older SMTP settings are complete; else None."""
    s = get_settings()
    if not _from_address():
        return None
    if s.brevo_api_key:
        return "brevo"
    if s.recovery_smtp_host and s.recovery_smtp_username and s.recovery_smtp_password:
        return "smtp"
    return None


def email_configured() -> bool:
    return email_provider() is not None


async def _send_brevo(to_email: str, to_name: str, subject: str, html: str, text: str, idempotency_key: Optional[str]) -> bool:
    s = get_settings()
    payload = {
        "sender": {"email": _from_address(), "name": s.email_from_name or "StudyMind"},
        "to": [{"email": to_email, **({"name": to_name} if to_name else {})}],
        "subject": subject,
        "htmlContent": html,
        "textContent": text,
    }
    if idempotency_key:
        payload["headers"] = {"idempotencyKey": idempotency_key}
    headers = {"accept": "application/json", "api-key": s.brevo_api_key, "content-type": "application/json"}
    for attempt in range(_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(BREVO_ENDPOINT, json=payload, headers=headers)
            if resp.status_code < 300:
                return True
            if resp.status_code < 500:                      # a client error will not improve by retrying
                log.error("email_rejected", status=resp.status_code)
                return False
            log.warning("email_provider_error", status=resp.status_code, attempt=attempt)
        except Exception as exc:                            # network / timeout
            log.warning("email_network_error", error_type=type(exc).__name__, attempt=attempt)
        if attempt < _RETRIES:
            await asyncio.sleep(1.0 * (attempt + 1))
    log.error("email_failed_after_retries")
    return False


def _send_smtp(to_email: str, subject: str, html: str, text: str) -> None:
    s = get_settings()
    message = EmailMessage()
    message["Subject"], message["From"], message["To"] = subject, _from_address(), to_email
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    with smtplib.SMTP(s.recovery_smtp_host, s.recovery_smtp_port, timeout=10) as smtp:
        smtp.starttls()
        smtp.login(s.recovery_smtp_username, s.recovery_smtp_password)
        smtp.send_message(message)


async def send_email(to_email: str, to_name: str, subject: str, html: str, text: str, idempotency_key: Optional[str] = None) -> bool:
    """True when the provider accepted the message. Never raises: a failed email must not break sign-up or recovery."""
    provider = email_provider()
    if provider is None:
        log.info("email_skipped_not_configured")
        return False
    try:
        if provider == "brevo":
            return await _send_brevo(to_email, to_name, subject, html, text, idempotency_key)
        await asyncio.to_thread(_send_smtp, to_email, subject, html, text)
        return True
    except Exception as exc:
        log.error("email_failed", error_type=type(exc).__name__)
        return False
