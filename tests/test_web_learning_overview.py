"""Read-only Web integration overview endpoint contract tests."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.auth import router
from app.auth.dependencies import require_auth
from app.db.engine import get_db


@pytest.mark.asyncio
async def test_learning_overview_returns_only_current_user_summary():
    user = SimpleNamespace(id=uuid4(), email="learner@example.test", full_name="Test Learner", role="self_learner")
    db = SimpleNamespace(scalar=AsyncMock(side_effect=[2, 1]))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/auth/learning-overview")
    assert response.status_code == 200
    assert response.json() == {
        "user": {"id": str(user.id), "email": user.email, "full_name": user.full_name, "role": user.role},
        "modules": {"owned": 2, "enrolled": 1},
    }
    assert db.scalar.await_count == 2


def test_learning_overview_requires_authentication():
    # The route uses the same require_auth dependency as the production /me endpoint.
    route = next(r for r in router.routes if getattr(r, "path", "") == "/api/auth/learning-overview")
    dependencies = [d.call for d in route.dependant.dependencies]
    assert require_auth in dependencies
