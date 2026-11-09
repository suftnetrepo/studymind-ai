"""Test-mode payment reversal reconciliation. No changes to production by default."""
from __future__ import annotations

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import StudyMindEntitlement
from app.billing.stripe_webhook import InvalidStripeWebhook


async def retrieve_payment_intent(payment_intent_id: str) -> dict:
    if not isinstance(payment_intent_id, str) or not payment_intent_id.startswith("pi_"):
        raise InvalidStripeWebhook("Invalid payment intent reference")
    settings = get_settings()
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"https://api.stripe.com/v1/payment_intents/{payment_intent_id}",
            params={"expand[]": "latest_charge"},
            headers={"Authorization": f"Bearer {settings.studymind_stripe_secret_key}"},
        )
    response.raise_for_status()
    intent = response.json()
    if intent.get("id") != payment_intent_id or intent.get("livemode") is not False:
        raise InvalidStripeWebhook("Invalid test-mode payment intent")
    return intent


def payment_reversal_status(intent: dict) -> str | None:
    """Use Stripe's current charge state, not an untrusted event payload.

    Full refund revokes. Partial refund suspends pending manual review.
    Any open or lost dispute suspends; won dispute may restore if not refunded.
    """
    charge = intent.get("latest_charge")
    if not isinstance(charge, dict):
        raise InvalidStripeWebhook("Charge details unavailable")
    if charge.get("livemode") is not False or charge.get("payment_intent") != intent.get("id"):
        raise InvalidStripeWebhook("Charge does not match payment")
    amount = charge.get("amount")
    refunded = charge.get("amount_refunded")
    if type(amount) is not int or amount <= 0 or type(refunded) is not int or not 0 <= refunded <= amount:
        raise InvalidStripeWebhook("Invalid refund amounts")
    if refunded == amount or charge.get("refunded") is True:
        return "revoked"
    if refunded > 0:
        return "suspended"
    if charge.get("disputed") is True:
        return "suspended"
    return "active"


async def reconcile_payment_reversal(db: AsyncSession, intent: dict) -> int:
    """Update only matching Stripe-issued entitlements. Caller commits."""
    status = payment_reversal_status(intent)
    rows = (await db.execute(
        select(StudyMindEntitlement).where(
            StudyMindEntitlement.stripe_payment_intent_id == intent["id"]
        ).with_for_update()
    )).scalars().all()
    for entitlement in rows:
        # A revoked entitlement is terminal; never restore it on later events.
        if entitlement.status == "revoked":
            continue
        entitlement.status = status
    await db.flush()
    return len(rows)
