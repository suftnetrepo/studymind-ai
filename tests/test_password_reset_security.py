"""Regression guard: unauthenticated password reset must never mutate credentials."""
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.auth import router


@pytest.mark.asyncio
async def test_legacy_password_reset_fails_closed_without_database_access():
    app = FastAPI()
    app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/auth/forgot-password",
            json={"email": "existing@example.test", "new_password": "NewSecurePass123"},
        )
    assert response.status_code == 410
    assert "temporarily unavailable" in response.json()["detail"].lower()


def test_legacy_reset_endpoint_cannot_write_password_hash():
    from inspect import getsource
    from app.api.auth import forgot_password
    source = getsource(forgot_password)
    assert "password_hash =" not in source
    assert "db.commit(" not in source
