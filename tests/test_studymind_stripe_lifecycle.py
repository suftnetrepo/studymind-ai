"""Refund/dispute state validation without live Stripe calls."""
import pytest
from app.billing.stripe_lifecycle import payment_reversal_status
from app.billing.stripe_webhook import InvalidStripeWebhook


def charge(refunded=0, disputed=False, amount=399):
    return {"id": "pi_test", "latest_charge": {"livemode": False,
            "payment_intent": "pi_test", "amount": amount,
            "amount_refunded": refunded, "refunded": refunded == amount,
            "disputed": disputed}}


def test_clean_payment_active():
    assert payment_reversal_status(charge()) == "active"


def test_partial_refund_suspends():
    assert payment_reversal_status(charge(refunded=100)) == "suspended"


def test_full_refund_revokes():
    assert payment_reversal_status(charge(refunded=399)) == "revoked"


def test_dispute_suspends():
    assert payment_reversal_status(charge(disputed=True)) == "suspended"


@pytest.mark.parametrize("bad", [
    {"id": "pi_test", "latest_charge": None},
    charge(refunded=500), charge(refunded=-1), charge(amount=0),
    {"id": "pi_test", "latest_charge": {"livemode": True, "payment_intent": "pi_test",
       "amount": 399, "amount_refunded": 0}},
])
def test_invalid_charge_rejected(bad):
    with pytest.raises(InvalidStripeWebhook):
        payment_reversal_status(bad)
