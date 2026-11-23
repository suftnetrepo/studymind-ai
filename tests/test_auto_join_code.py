"""Class modules get a short, unique ``CIQ-XXXX`` join code that the join endpoint accepts."""
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.api import modules
from app.db.schemas import ModuleCreate

CODE = re.compile(r"^CIQ-[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{4}$")


def _user(role="lecturer"):
    return SimpleNamespace(id=uuid4(), role=role, institution_id=None, department_id=None)


class FakeDb:
    """Just enough of AsyncSession: stores added modules and answers course_code lookups."""

    def __init__(self, taken=()):
        self.modules = [SimpleNamespace(id=uuid4(), course_code=c, access_type="class", status="active") for c in taken]
        self.queries = 0
        self.add = MagicMock(side_effect=lambda m: self.modules.append(m))
        self.commit, self.refresh = AsyncMock(), AsyncMock()

    async def execute(self, stmt):
        self.queries += 1
        sql = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
        wanted = re.search(r"course_code = '([^']+)'", sql)
        hits = [m for m in self.modules if wanted and m.course_code == wanted.group(1)]
        result = MagicMock()
        result.first.return_value = (hits[0].id,) if hits else None
        result.scalars.return_value.limit.return_value = result.scalars.return_value
        result.scalars.return_value.all.return_value = [m.id for m in hits]
        return result


@pytest.fixture(autouse=True)
def _plain_schema(monkeypatch):
    monkeypatch.setattr(modules.ModuleSchema, "model_validate", staticmethod(lambda m: SimpleNamespace(
        document_count=0, student_count=0, emoji=None, course_code=m.course_code, id=m.id)))


async def _create(db, **kw):
    return await modules.create_module(ModuleCreate(title="Algorithms", access_type="class", **kw), db, _user())


@pytest.mark.asyncio
async def test_class_module_without_code_gets_ciq_code():
    out = await _create(FakeDb())
    assert CODE.match(out.course_code)


@pytest.mark.asyncio
async def test_explicit_code_is_kept_and_uppercased():
    out = await _create(FakeDb(), course_code="csc109")
    assert out.course_code == "CSC109"


@pytest.mark.asyncio
async def test_personal_module_gets_no_code():
    out = await modules.create_module(ModuleCreate(title="My notes", access_type="personal"), FakeDb(), _user("student"))
    assert out.course_code is None


@pytest.mark.asyncio
async def test_codes_are_unique_across_modules():
    db = FakeDb()
    codes = {(await _create(db)).course_code for _ in range(50)}
    assert len(codes) == 50


@pytest.mark.asyncio
async def test_collision_is_retried(monkeypatch):
    # first candidate is taken (CIQ-AAAA), the second is free (CIQ-BBBB)
    picks = iter(list("AAAA") + list("BBBB"))
    monkeypatch.setattr(modules.secrets, "choice", lambda _alphabet: next(picks))
    db = FakeDb(taken=["CIQ-AAAA"])
    out = await _create(db)
    assert out.course_code == "CIQ-BBBB"


@pytest.mark.asyncio
async def test_gives_up_with_500_after_ten_collisions(monkeypatch):
    monkeypatch.setattr(modules.secrets, "choice", lambda _alphabet: "A")
    db = FakeDb(taken=["CIQ-AAAA"])
    with pytest.raises(HTTPException) as exc:
        await _create(db)
    assert exc.value.status_code == 500
    assert db.queries == 10
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_student_joins_with_the_generated_code(monkeypatch):
    db = FakeDb()
    created = await _create(db)
    module = db.modules[0]
    lookups = []

    async def execute(stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
        lookups.append(sql)
        result = MagicMock()
        result.scalars.return_value.all.return_value = [module.id] if f"course_code = '{created.course_code}'" in sql else []
        return result

    enrolled = AsyncMock(return_value=SimpleNamespace(module_id=module.id, id=uuid4()))
    monkeypatch.setattr(modules, "enrol_in_module", enrolled)
    join_db = MagicMock(execute=execute)

    # students type codes in any case
    out = await modules.join_module_by_course_id(modules.JoinModuleRequest(course_id=created.course_code.lower()), join_db, _user("student"))

    assert out["module_id"] == str(module.id)
    assert enrolled.await_args.args[0] == module.id
    assert any(f"course_code = '{created.course_code}'" in q for q in lookups)


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", [None, ""])
async def test_editing_a_class_module_with_a_blank_code_keeps_its_join_code(monkeypatch, blank):
    module = SimpleNamespace(id=uuid4(), course_code="CIQ-K7M2", access_type="class", status="active")
    monkeypatch.setattr(modules, "_get_owned_module", AsyncMock(return_value=module))
    db = FakeDb()
    await modules.update_module(module.id, ModuleCreate(title="Renamed", access_type="class", course_code=blank), db, _user())
    assert module.course_code == "CIQ-K7M2"


@pytest.mark.asyncio
async def test_editing_a_legacy_class_module_without_a_code_gives_it_one(monkeypatch):
    module = SimpleNamespace(id=uuid4(), course_code=None, access_type="class", status="active")
    monkeypatch.setattr(modules, "_get_owned_module", AsyncMock(return_value=module))
    await modules.update_module(module.id, ModuleCreate(title="Renamed", access_type="class"), FakeDb(), _user())
    assert CODE.match(module.course_code)
