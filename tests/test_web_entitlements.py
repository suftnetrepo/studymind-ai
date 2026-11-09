"""Canonical entitlements: no client-side activation or unauthenticated reads."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.api.entitlements import my_entitlements, router


class _Scalars:
    def __init__(self, rows): self.rows = rows
    def scalars(self): return self
    def all(self): return self.rows


@pytest.mark.asyncio
async def test_free_user_entitlement_response():
    db = SimpleNamespace(execute=AsyncMock(return_value=_Scalars([])))
    response = await my_entitlements(db, SimpleNamespace(id=uuid4()))
    assert response["plan"] == "free"
    assert response["features"]["chat"] == {"limit": 10, "used": 0, "remaining": 10}
    assert response["features"]["quiz"]["limit"] == 2
    assert response["enforcement_enabled"] is False
    assert db.execute.await_count == 2


def test_entitlement_endpoint_read_only():
    routes = [r for r in router.routes if r.path == "/api/me/entitlements"]
    assert len(routes) == 1
    assert routes[0].methods == {"GET"}


def test_entitlement_schema_is_bound_to_canonical_user():
    from app.db.models import StudyMindEntitlement
    assert "user_id" in StudyMindEntitlement.__table__.columns
    assert "stripe_checkout_session_id" in StudyMindEntitlement.__table__.columns
