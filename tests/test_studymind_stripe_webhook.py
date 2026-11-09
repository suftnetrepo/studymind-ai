"""Pure webhook validation tests; no live Stripe access or database required."""
import hashlib
import hmac
import json
from uuid import uuid4

import pytest

from app.billing.stripe_webhook import (
    InvalidStripeWebhook, validate_paid_checkout, verify_stripe_signature,
)


def signed(body: bytes, timestamp: int = 1700000000):
    digest = hmac.new(b"whsec_test", str(timestamp).encode() + b"." + body, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def test_valid_signed_event():
    body = json.dumps({"type": "checkout.session.completed"}).encode()
    assert verify_stripe_signature(body, signed(body), "whsec_test", now=1700000000)["type"] == "checkout.session.completed"


@pytest.mark.parametrize("header,body", [
    ("", b"{}"),
    ("t=1700000000,v1=bad", b'{"type":"x"}'),
    (signed(b'{"type":"x"}'), b'{"type":"y"}'),
])
def test_reject_invalid_signature(header, body):
    with pytest.raises(InvalidStripeWebhook):
        verify_stripe_signature(body, header, "whsec_test", now=1700000000)


def test_reject_stale_signature():
    body = b'{"type":"x"}'
    with pytest.raises(InvalidStripeWebhook):
        verify_stripe_signature(body, signed(body), "whsec_test", now=1700000400)


def valid_session():
    user_id = str(uuid4())
    return {"id": "cs_test_123", "livemode": False, "mode": "payment",
            "payment_status": "paid", "status": "complete", "currency": "gbp",
            "amount_total": 399, "client_reference_id": user_id,
            "payment_intent": "pi_test_123",
            "metadata": {"studymind_user_id": user_id, "studymind_product": "plus_28d_v1"}}


def test_valid_checkout():
    session = valid_session()
    user_id, payment_intent = validate_paid_checkout(session)
    assert str(user_id) == session["client_reference_id"]
    assert payment_intent == "pi_test_123"


@pytest.mark.parametrize("field,value", [
    ("livemode", True), ("payment_status", "unpaid"), ("status", "open"),
    ("currency", "usd"), ("amount_total", 0), ("mode", "subscription"),
    ("payment_intent", None),
])
def test_reject_ineligible_checkout(field, value):
    session = valid_session()
    session[field] = value
    with pytest.raises(InvalidStripeWebhook):
        validate_paid_checkout(session)


def test_reject_mismatched_user():
    session = valid_session()
    session["client_reference_id"] = str(uuid4())
    with pytest.raises(InvalidStripeWebhook):
        validate_paid_checkout(session)
