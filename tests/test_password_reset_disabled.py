"""Legacy unauthenticated password replacement must never return to the live API."""
import pytest
from fastapi import HTTPException
from app.api.auth import forgot_password
from app.db.schemas import ForgotPasswordRequest


@pytest.mark.asyncio
async def test_legacy_password_reset_fails_closed_without_database():
    # No DB dependency or password mutation is allowed on this endpoint.
    with pytest.raises(HTTPException) as exc:
        await forgot_password(body=ForgotPasswordRequest(email="student@example.com", new_password="ValidPassword123!"), request=None)
    assert exc.value.status_code == 410
