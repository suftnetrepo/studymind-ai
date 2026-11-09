"""Checkout foundation checks; no network or real Stripe credentials needed."""
from unittest.mock import patch
import pytest
from fastapi import HTTPException
from app.config import Settings
from app.billing.stripe_checkout import checkout_configured
from app.api.billing import checkout


def test_billing_off_by_default():
    assert Settings.model_fields["studymind_billing_enabled"].default is False


def test_checkout_requires_complete_configuration():
    with patch("app.billing.stripe_checkout.get_settings") as get_settings:
        s = get_settings.return_value
        s.studymind_billing_enabled = False
        s.studymind_stripe_secret_key = "sk_test_fake"
        s.studymind_stripe_webhook_secret = "whsec_fake"
        s.studymind_checkout_success_url = "https://example.com/success"
        s.studymind_checkout_cancel_url = "https://example.com/cancel"
        assert not checkout_configured()
        s.studymind_billing_enabled = True
        assert checkout_configured()
        s.studymind_stripe_secret_key = "sk_live_fake"
        assert not checkout_configured()


@pytest.mark.asyncio
async def test_checkout_fails_closed_without_config():
    with patch("app.api.billing.checkout_configured", return_value=False):
        with pytest.raises(HTTPException) as exc:
            await checkout(None)
    assert exc.value.status_code == 503
