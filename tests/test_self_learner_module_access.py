"""Enrolled self-learners must have the same class-read access as students."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from app.api import modules


@pytest.mark.asyncio
async def test_enrolled_self_learner_can_read_class(monkeypatch):
    owner_id, learner_id, module_id = uuid4(), uuid4(), uuid4()
    module = SimpleNamespace(id=module_id, owner_id=owner_id, access_type="class", institution_id=None)
    monkeypatch.setattr(modules, "_get_module_or_404", AsyncMock(return_value=module))
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: object())))
    user = SimpleNamespace(id=learner_id, role="self_learner", institution_id=None)
    assert await modules._get_accessible_module(module_id, user, db) is module
    db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_unenrolled_self_learner_cannot_read_class(monkeypatch):
    from fastapi import HTTPException
    owner_id, learner_id, module_id = uuid4(), uuid4(), uuid4()
    module = SimpleNamespace(id=module_id, owner_id=owner_id, access_type="class", institution_id=None)
    monkeypatch.setattr(modules, "_get_module_or_404", AsyncMock(return_value=module))
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None)))
    user = SimpleNamespace(id=learner_id, role="self_learner", institution_id=None)
    with pytest.raises(HTTPException) as exc:
        await modules._get_accessible_module(module_id, user, db)
    assert exc.value.status_code == 403
