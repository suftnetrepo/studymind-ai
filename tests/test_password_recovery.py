"""Password recovery must be off by default and use single-use hashed tokens."""
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from app.auth.recovery import new_recovery_token, hash_recovery_token
from app.api.auth import RecoveryConfirm, RecoveryRequest, confirm_password_recovery, request_password_recovery
from app.config import Settings


def test_recovery_is_disabled_by_default():
    assert Settings.model_fields["password_recovery_enabled"].default is False


def test_recovery_token_is_random_and_only_hash_is_stored():
    a, digest_a = new_recovery_token()
    b, digest_b = new_recovery_token()
    assert a != b and digest_a != digest_b
    assert digest_a == hash_recovery_token(a)
    assert len(digest_a) == 64
    assert a not in digest_a


@pytest.mark.asyncio
async def test_request_fails_closed_without_smtp():
    with patch("app.api.auth.recovery_configured", return_value=False):
        with pytest.raises(HTTPException) as exc:
            await request_password_recovery(RecoveryRequest(email="student@example.com"), None, None)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_confirm_fails_closed_without_smtp():
    with patch("app.api.auth.recovery_configured", return_value=False):
        with pytest.raises(HTTPException) as exc:
            await confirm_password_recovery(RecoveryConfirm(token="a" * 48, new_password="Password123"), None)
    assert exc.value.status_code == 503
