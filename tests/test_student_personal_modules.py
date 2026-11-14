"""Students get the same personal-module privileges as self-learners; class modules stay with lecturers."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api import modules
from app.db.schemas import ModuleCreate


def _user(role):
    return SimpleNamespace(id=uuid4(), role=role, institution_id=None, department_id=None)


def _db():
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["student", "self_learner"])
async def test_student_and_self_learner_can_create_a_personal_module(role, monkeypatch):
    monkeypatch.setattr(modules.ModuleSchema, "model_validate", staticmethod(lambda m: SimpleNamespace(document_count=0, student_count=0, emoji=None)))
    db = _db()
    out = await modules.create_module(ModuleCreate(title="My notes", access_type="personal"), db, _user(role))
    assert out is not None
    db.add.assert_called_once()
    assert db.add.call_args[0][0].access_type == "personal"


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["student", "self_learner"])
async def test_student_and_self_learner_cannot_create_a_class_module(role):
    db = _db()
    with pytest.raises(HTTPException) as exc:
        await modules.create_module(ModuleCreate(title="Class", access_type="class"), db, _user(role))
    assert exc.value.status_code == 403
    db.add.assert_not_called()
