"""Stripe test-mode webhook verification and idempotent paid Checkout reconciliation.

No browser redirect can grant an entitlement. Billing stays disabled by default.
Refunds/disputes require a separate lifecycle handler before rollout.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import StudyMindEntitlement
from app.usage.studymind_quota import PLUS_DURATION_DAYS, PLUS_LIMITS, PLUS_PRICE_GBP_PENCE


class InvalidStripeWebhook(ValueError):
    pass


def verify_stripe_signature(body: bytes, header: str, secret: str, *, now: int | None = None) -> dict:
    """Verify Stripe's timestamped v1 HMAC against the exact raw request bytes."""
    parts: dict[str, list[str]] = {}
    for part in header.split(","):
        key, sep, value = part.strip().partition("=")
        if sep:
            parts.setdefault(key, []).append(value)
    try:
        timestamp = int(parts["t"][0])
    except (KeyError, ValueError, IndexError) as exc:
        raise InvalidStripeWebhook("Invalid Stripe signature") from exc
    current = int(time.time()) if now is None else now
    if abs(current - timestamp) > 300 or timestamp <= 0:
        raise InvalidStripeWebhook("Stripe signature timestamp outside tolerance")
    if not secret.startswith("whsec_"):
        raise InvalidStripeWebhook("Stripe webhook secret not configured")
    expected = hmac.new(secret.encode(), str(timestamp).encode() + b"." + body, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, candidate) for candidate in parts.get("v1", [])):
        raise InvalidStripeWebhook("Invalid Stripe signature")
    try:
        event = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidStripeWebhook("Invalid Stripe event") from exc
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise InvalidStripeWebhook("Invalid Stripe event")
    return event


async def retrieve_paid_checkout(session_id: str) -> dict:
    """Re-read the authoritative Checkout Session, never trust webhook fields alone."""
    if not session_id.startswith("cs_test_"):
        raise InvalidStripeWebhook("Only test-mode sessions are supported")
    settings = get_settings()
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"https://api.stripe.com/v1/checkout/sessions/{session_id}",
            headers={"Authorization": f"Bearer {settings.studymind_stripe_secret_key}"},
        )
    response.raise_for_status()
    return response.json()


def validate_paid_checkout(session: dict) -> tuple[UUID, str]:
    metadata = session.get("metadata") or {}
    user_ref = metadata.get("studymind_user_id")
    if (session.get("livemode") is not False
        or session.get("mode") != "payment"
        or session.get("payment_status") != "paid"
        or session.get("status") != "complete"
        or session.get("currency") != "gbp"
        or session.get("amount_total") != PLUS_PRICE_GBP_PENCE
        or metadata.get("studymind_product") != "plus_28d_v1"
        or session.get("client_reference_id") != user_ref
        or not isinstance(session.get("payment_intent"), str)):
        raise InvalidStripeWebhook("Checkout is not an eligible paid StudyMind pass")
    try:
        return UUID(user_ref), session["payment_intent"]
    except (TypeError, ValueError, AttributeError) as exc:
        raise InvalidStripeWebhook("Invalid Checkout user reference") from exc


async def activate_paid_checkout(db: AsyncSession, session: dict) -> bool:
    """Insert exactly once per Stripe Checkout Session; caller commits transaction."""
    user_id, payment_intent = validate_paid_checkout(session)
    session_id = session.get("id")
    if not isinstance(session_id, str) or not session_id.startswith("cs_test_"):
        raise InvalidStripeWebhook("Invalid Checkout Session ID")
    now = datetime.now(timezone.utc)
    statement = pg_insert(StudyMindEntitlement).values(
        id=uuid4(), user_id=user_id, stripe_checkout_session_id=session_id,
        stripe_payment_intent_id=payment_intent, status="active",
        feature_limits=dict(PLUS_LIMITS), starts_at=now,
        expires_at=now + timedelta(days=PLUS_DURATION_DAYS),
    ).on_conflict_do_nothing(index_elements=["stripe_checkout_session_id"])
    result = await db.execute(statement)
    return result.rowcount == 1
