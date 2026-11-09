"""Additive Web join endpoint contract tests; no production database needed."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.modules import JoinModuleRequest, join_module_by_course_id


@pytest.mark.asyncio
async def test_join_rejects_lecturer_without_database_access():
    db = SimpleNamespace(execute=AsyncMock())
    with pytest.raises(HTTPException) as exc:
        await join_module_by_course_id(
            JoinModuleRequest(course_id=str(uuid4())), db, SimpleNamespace(role="lecturer")
        )
    assert exc.value.status_code == 403
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_join_rejects_invalid_course_id_without_database_access():
    db = SimpleNamespace(execute=AsyncMock())
    with pytest.raises(HTTPException) as exc:
        await join_module_by_course_id(
            JoinModuleRequest(course_id="invalid code!"), db, SimpleNamespace(role="student")
        )
    assert exc.value.status_code == 422
    db.execute.assert_not_awaited()


def test_join_endpoint_is_registered():
    from app.api.modules import router
    assert any(getattr(route, "path", None) == "/api/modules/join" and "POST" in route.methods for route in router.routes)
