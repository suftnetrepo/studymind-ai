"""Enrolment safety endpoints (preview before joining, leave, lecturer removes a student): behaviour and authorization.

Same style as the other route tests: a tiny FastAPI app, the real router and the real role gate, a mocked database session.
`require_auth` is `get_current_user`, so overriding it with a test user lets the real role checks run.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.modules import router
from app.auth.dependencies import require_auth
from app.db.engine import get_db


def user(role, **kw):
    return SimpleNamespace(id=uuid4(), role=role, email=f"{role}@example.test", full_name=kw.pop("full_name", role.title()), is_active=True, **kw)


def module(owner_id, **kw):
    base = dict(id=uuid4(), title="Year 5 Mathematics", description=None, status="active", access_type="class", owner_id=owner_id)
    base.update(kw)
    return SimpleNamespace(**base)


def one(value):      # result of `(await db.execute(...)).scalar_one_or_none()`
    return SimpleNamespace(scalar_one_or_none=lambda: value)


def many(values):    # result of `(await db.execute(...)).scalars().all()`
    return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: list(values)))


def make_db(*results, get=None):
    return SimpleNamespace(execute=AsyncMock(side_effect=list(results)), get=AsyncMock(return_value=get), delete=AsyncMock(), commit=AsyncMock())


async def request(method, url, caller, db):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = lambda: caller
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.request(method, url)


# ── GET /api/modules/join/preview ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_preview_shows_title_lecturer_and_status_before_joining():
    lecturer, student = user("lecturer", full_name="Abel Dama"), user("student")
    m = module(lecturer.id)
    db = make_db(one(m), one(None), get=lecturer)                      # module by UUID, then "am I enrolled?"
    r = await request("GET", f"/api/modules/join/preview?course_id={m.id}", student, db)
    assert r.status_code == 200
    assert r.json() == {"module_id": str(m.id), "title": "Year 5 Mathematics", "description": None, "lecturer_name": "Abel Dama", "status": "active", "already_enrolled": False}
    db.delete.assert_not_awaited(); db.commit.assert_not_awaited()      # a preview never changes anything


@pytest.mark.asyncio
async def test_preview_reports_when_the_student_is_already_enrolled():
    lecturer, student = user("lecturer"), user("self_learner")
    m = module(lecturer.id)
    db = make_db(one(m), one(SimpleNamespace(id=uuid4())), get=lecturer)
    r = await request("GET", f"/api/modules/join/preview?course_id={m.id}", student, db)
    assert r.status_code == 200 and r.json()["already_enrolled"] is True


@pytest.mark.asyncio
async def test_preview_of_an_unknown_id_is_a_clean_404():
    db = make_db(one(None), many([]), one(None))                       # not a module UUID, no course code, no enrolment code
    r = await request("GET", f"/api/modules/join/preview?course_id={uuid4()}", user("student"), db)
    assert r.status_code == 404 and r.json()["detail"] == "Course ID not found"


@pytest.mark.asyncio
async def test_preview_of_an_ambiguous_course_code_asks_for_the_uuid():
    lecturer = user("lecturer")
    db = make_db(many([module(lecturer.id), module(lecturer.id)]))
    r = await request("GET", "/api/modules/join/preview?course_id=MATH5", user("student"), db)
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_preview_is_for_students_only_and_blank_ids_are_rejected():
    r = await request("GET", f"/api/modules/join/preview?course_id={uuid4()}", user("lecturer"), make_db())
    assert r.status_code == 403
    r = await request("GET", "/api/modules/join/preview?course_id=%20%20", user("student"), make_db())
    assert r.status_code == 422


# ── DELETE /api/modules/{id}/leave ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_student_can_leave_a_module_they_are_in():
    student, enrolment = user("student"), SimpleNamespace(id=uuid4())
    db = make_db(one(enrolment))
    r = await request("DELETE", f"/api/modules/{uuid4()}/leave", student, db)
    assert r.status_code == 204
    db.delete.assert_awaited_once_with(enrolment); db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_leaving_a_module_you_are_not_in_is_404_and_deletes_nothing():
    db = make_db(one(None))
    r = await request("DELETE", f"/api/modules/{uuid4()}/leave", user("student"), db)
    assert r.status_code == 404
    db.delete.assert_not_awaited(); db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_students_can_leave():
    db = make_db()
    r = await request("DELETE", f"/api/modules/{uuid4()}/leave", user("lecturer"), db)
    assert r.status_code == 403
    db.execute.assert_not_awaited()


# ── DELETE /api/modules/{id}/students/{student_id} ──────────────────────────────

@pytest.mark.asyncio
async def test_owner_lecturer_can_remove_an_enrolled_student():
    lecturer, enrolment = user("lecturer"), SimpleNamespace(id=uuid4())
    m = module(lecturer.id)
    db = make_db(one(m), one(enrolment))                                # owned-module lookup, then the enrolment
    r = await request("DELETE", f"/api/modules/{m.id}/students/{uuid4()}", lecturer, db)
    assert r.status_code == 204
    db.delete.assert_awaited_once_with(enrolment); db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_lecturer_cannot_remove_students_from_someone_elses_module():
    lecturer = user("lecturer")
    m = module(uuid4())                                                 # owned by another user
    db = make_db(one(m))
    r = await request("DELETE", f"/api/modules/{m.id}/students/{uuid4()}", lecturer, db)
    assert r.status_code == 403
    db.delete.assert_not_awaited(); db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_student_cannot_remove_anyone_the_role_gate_runs_before_the_database():
    db = make_db()
    r = await request("DELETE", f"/api/modules/{uuid4()}/students/{uuid4()}", user("student"), db)
    assert r.status_code == 403
    db.execute.assert_not_awaited(); db.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_removing_a_student_who_is_not_enrolled_is_404():
    lecturer = user("lecturer")
    m = module(lecturer.id)
    db = make_db(one(m), one(None))
    r = await request("DELETE", f"/api/modules/{m.id}/students/{uuid4()}", lecturer, db)
    assert r.status_code == 404
    db.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_admin_may_remove_a_student_from_any_module():
    admin, enrolment = user("admin"), SimpleNamespace(id=uuid4())
    m = module(uuid4())
    db = make_db(one(m), one(enrolment))
    r = await request("DELETE", f"/api/modules/{m.id}/students/{uuid4()}", admin, db)
    assert r.status_code == 204
    db.delete.assert_awaited_once_with(enrolment)
