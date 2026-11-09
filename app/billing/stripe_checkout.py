"""Web-only, opt-in Stripe Checkout creation for the StudyMind 28-day pass."""
from __future__ import annotations

from urllib.parse import urlsplit
import httpx
from app.config import get_settings
from app.usage.studymind_quota import PLUS_PRICE_GBP_PENCE


def checkout_configured() -> bool:
    s = get_settings()
    urls = (s.studymind_checkout_success_url, s.studymind_checkout_cancel_url)
    return bool(s.studymind_billing_enabled and s.studymind_stripe_secret_key.startswith("sk_test_")
                and s.studymind_stripe_webhook_secret.startswith("whsec_")
                and all(urlsplit(url).scheme == "https" and urlsplit(url).netloc for url in urls))


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
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            "https://api.stripe.com/v1/checkout/sessions",
            data=data,
            headers={"Authorization": f"Bearer {s.studymind_stripe_secret_key}"},
        )
    response.raise_for_status()
    result = response.json()
    if not result.get("id") or not result.get("url"):
        raise RuntimeError("Stripe did not return a Checkout Session")
    return {"checkout_session_id": result["id"], "checkout_url": result["url"]}
