"""StudyMind Web-only Checkout endpoint; never activates access from the browser."""
from fastapi import APIRouter, Depends, HTTPException, Request
from app.auth.dependencies import require_auth
from app.db.models import User
from app.db.engine import AsyncSessionLocal
from app.config import get_settings
from app.billing.stripe_webhook import (
    InvalidStripeWebhook, verify_stripe_signature, retrieve_paid_checkout, activate_paid_checkout,
)
from app.billing.stripe_checkout import checkout_configured, create_checkout
from app.billing.stripe_lifecycle import retrieve_payment_intent, reconcile_payment_reversal
from app.logging_config import get_logger

router = APIRouter(prefix="/api/billing", tags=["billing"])
log = get_logger(__name__)


@router.post("/checkout")
async def checkout(current_user: User = Depends(require_auth)):
    if not checkout_configured():
        raise HTTPException(status_code=503, detail="StudyMind Plus checkout is not available yet.")
    try:
        return await create_checkout(str(current_user.id), current_user.email)
    except Exception as exc:
        log.error("studymind_checkout_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Unable to start checkout. Please try again.") from None


@router.post("/stripe/webhook", status_code=200)
async def stripe_webhook(request: Request):
    settings = get_settings()
    if not checkout_configured():
        raise HTTPException(status_code=503, detail="Billing is disabled")
    body = await request.body()
    if len(body) > 65536:
        raise HTTPException(status_code=413, detail="Webhook payload too large")
    try:
        event = verify_stripe_signature(
            body, request.headers.get("stripe-signature", ""), settings.studymind_stripe_webhook_secret
        )
    except InvalidStripeWebhook:
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook") from None

    reversal_events = {
        "charge.refunded", "charge.dispute.created", "charge.dispute.updated",
        "charge.dispute.closed", "charge.dispute.funds_withdrawn",
        "charge.dispute.funds_reinstated",
    }
    if event["type"] in reversal_events:
        obj = (event.get("data") or {}).get("object") or {}
        payment_intent_id = obj.get("payment_intent")
        if not isinstance(payment_intent_id, str) or not payment_intent_id.startswith("pi_"):
            raise HTTPException(status_code=400, detail="Invalid payment reference")
        try:
            intent = await retrieve_payment_intent(payment_intent_id)
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    updated = await reconcile_payment_reversal(db, intent)
        except InvalidStripeWebhook:
            raise HTTPException(status_code=400, detail="Payment verification failed") from None
        except Exception as exc:
            log.error("studymind_reversal_failed", error_type=type(exc).__name__)
            raise HTTPException(status_code=503, detail="Payment reconciliation unavailable") from None
        return {"received": True, "reconciled": updated}

    if event["type"] != "checkout.session.completed":
        # Do not activate on refunds, disputes, or asynchronous unpaid events.
        return {"received": True, "activated": False}
    obj = (event.get("data") or {}).get("object") or {}
    session_id = obj.get("id")
    if not isinstance(session_id, str) or not session_id.startswith("cs_test_"):
        raise HTTPException(status_code=400, detail="Invalid Checkout reference")
    try:
        paid_session = await retrieve_paid_checkout(session_id)
        if paid_session.get("id") != session_id:
            raise InvalidStripeWebhook("Checkout Session mismatch")
        async with AsyncSessionLocal() as db:
            async with db.begin():
                activated = await activate_paid_checkout(db, paid_session)
    except InvalidStripeWebhook:
        raise HTTPException(status_code=400, detail="Checkout verification failed") from None
    except Exception as exc:
        log.error("studymind_webhook_processing_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=503, detail="Webhook processing temporarily unavailable") from None
    return {"received": True, "activated": activated}
