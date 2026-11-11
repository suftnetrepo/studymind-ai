"""Web-only, opt-in Stripe Checkout creation for the StudyMind 28-day pass."""
from __future__ import annotations

from urllib.parse import urlsplit
import httpx
from app.config import get_settings
from app.usage.studymind_quota import PLUS_PRICE_GBP_PENCE


STRIPE_API = "https://api.stripe.com"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_LOCAL_ENVS = {"development", "test"}


def stripe_api_base() -> str:
    """The Stripe API origin. The real API always; a loopback stand-in only outside staging and production."""
    s = get_settings()
    base = getattr(s, "studymind_stripe_api_base", STRIPE_API)
    base = (base if isinstance(base, str) and base else STRIPE_API).rstrip("/")
    if base == STRIPE_API:
        return base
    parts = urlsplit(base)
    if parts.scheme in ("http", "https") and parts.hostname in _LOOPBACK and s.app_env in _LOCAL_ENVS:
        return base
    raise RuntimeError("Stripe API base override refused (loopback and development/test only)")


def _return_url_ok(url: str, app_env: str) -> bool:
    parts = urlsplit(url)
    if not parts.netloc:
        return False
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and parts.hostname in _LOOPBACK and app_env in _LOCAL_ENVS   # local development only


def checkout_configured() -> bool:
    s = get_settings()
    urls = (s.studymind_checkout_success_url, s.studymind_checkout_cancel_url)
    try:
        stripe_api_base()
    except RuntimeError:
        return False
    return bool(s.studymind_billing_enabled and s.studymind_stripe_secret_key.startswith("sk_test_")
                and s.studymind_stripe_webhook_secret.startswith("whsec_")
                and all(_return_url_ok(url, s.app_env) for url in urls))


async def create_checkout(user_id: str, email: str) -> dict:
    """Only the authenticated backend supplies identity and fixed price."""
    s = get_settings()
    if not checkout_configured():
        raise RuntimeError("Billing is not configured")
    data = {
        "mode": "payment",
        "customer_email": email,
        "success_url": s.studymind_checkout_success_url,
        "cancel_url": s.studymind_checkout_cancel_url,
        "line_items[0][price_data][currency]": "gbp",
        "line_items[0][price_data][unit_amount]": str(PLUS_PRICE_GBP_PENCE),
        "line_items[0][price_data][product_data][name]": "StudyMind Plus — 28-day pass",
        "line_items[0][quantity]": "1",
        "client_reference_id": user_id,
        "metadata[studymind_user_id]": user_id,
        "metadata[studymind_product]": "plus_28d_v1",
        "payment_intent_data[metadata][studymind_user_id]": user_id,
    }
    tax_code = getattr(s, "studymind_stripe_tax_code", "")
    if isinstance(tax_code, str) and tax_code.startswith("txcd_"):
        data["line_items[0][price_data][product_data][tax_code]"] = tax_code
    if getattr(s, "studymind_stripe_managed_payments", True) is False:
        data["managed_payments[enabled]"] = "false"
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"{stripe_api_base()}/v1/checkout/sessions",
            data=data,
            headers={"Authorization": f"Bearer {s.studymind_stripe_secret_key}"},
        )
    response.raise_for_status()
    result = response.json()
    if not result.get("id") or not result.get("url"):
        raise RuntimeError("Stripe did not return a Checkout Session")
    return {"checkout_session_id": result["id"], "checkout_url": result["url"]}


async def has_active_plus(db, user_id) -> bool:
    """True when the user holds a currently active, unexpired Stripe-issued pass."""
    from datetime import datetime, timezone
    from sqlalchemy import select
    from app.db.models import StudyMindEntitlement
    now = datetime.now(timezone.utc)
    row = (await db.execute(select(StudyMindEntitlement.id).where(
        StudyMindEntitlement.user_id == user_id, StudyMindEntitlement.status == "active",
        StudyMindEntitlement.starts_at <= now, StudyMindEntitlement.expires_at > now).limit(1))).first()
    return row is not None
