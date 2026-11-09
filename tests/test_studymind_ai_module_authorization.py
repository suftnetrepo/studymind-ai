"""Request-level regression guard: every AI generation endpoint must authorize module access BEFORE it
creates a chat session, enters usage metering or calls the AI provider / retrieval layer.

These tests drive the real FastAPI app (routing, request validation, dependency wiring, the real
`_get_accessible_module` rules). Only authentication, the DB session and the AI boundary are replaced:
the AI boundary is a spy that records every call, so "the provider was never reached" is asserted directly
rather than inferred from source text. No database, network or OpenAI key is needed.
"""
import contextlib
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.api import features as features_api
from app.api import modules as modules_api
from app.api.main import app
from app.auth.dependencies import require_auth
from app.db.engine import get_db

OWNER = uuid.uuid4()
OTHER_OWNER = uuid.uuid4()
LEARNER = SimpleNamespace(id=uuid.uuid4(), role="self_learner", institution_id=None)

MY_MODULE = uuid.uuid4()            # LEARNER is enrolled
MY_WEEK = uuid.uuid4()
FOREIGN_MODULE = uuid.uuid4()       # LEARNER is not enrolled (someone else's class)
FOREIGN_WEEK = uuid.uuid4()
UNKNOWN = uuid.uuid4()

MODULES = {
    MY_MODULE: SimpleNamespace(id=MY_MODULE, owner_id=OWNER, access_type="class", institution_id=None),
    FOREIGN_MODULE: SimpleNamespace(id=FOREIGN_MODULE, owner_id=OTHER_OWNER, access_type="class", institution_id=None),
}
WEEKS = {MY_WEEK: MY_MODULE, FOREIGN_WEEK: FOREIGN_MODULE}
ENROLLED_IN = {MY_MODULE}


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class FakeDB:
    """Answers only the two lookups authorization performs; anything else fails the test loudly."""

    async def execute(self, stmt, *a, **kw):
        sql = str(stmt).lower()
        if "from weeks" in sql:
            wanted = [v for v in stmt.compile().params.values() if isinstance(v, uuid.UUID)]
            return _Result(WEEKS.get(wanted[0]) if wanted else None)
        if "from enrolments" in sql:
            wanted = [v for v in stmt.compile().params.values() if isinstance(v, uuid.UUID)]
            return _Result(object() if any(w in ENROLLED_IN for w in wanted) else None)
        raise AssertionError(f"unexpected DB access before authorization completed: {sql[:120]}")


class _ReachedProvider(Exception):
    """Raised by the spies so a request that WAS authorized stops at the AI boundary."""


@pytest.fixture
def api(monkeypatch):
    calls = SimpleNamespace(ai=[], metered=[], sessions=[])

    async def _module_or_404(module_id, db):
        if module_id not in MODULES:
            raise HTTPException(status_code=404, detail="Module not found")
        return MODULES[module_id]

    monkeypatch.setattr(modules_api, "_get_module_or_404", _module_or_404)

    @contextlib.asynccontextmanager
    async def metered(user_id, feature):
        calls.metered.append(feature)
        yield

    def spy(name):
        def _fn(*a, **kw):
            calls.ai.append(name)
            raise _ReachedProvider(name)
        return _fn

    for mod in (features_api, chat_api):
        monkeypatch.setattr(mod, "metered_generation", metered)
    monkeypatch.setattr(features_api, "generate_quiz", spy("quiz"))
    monkeypatch.setattr(features_api, "generate_flashcards", spy("flashcards"))
    monkeypatch.setattr(features_api, "generate_summary", spy("summary"))

    async def session(*a, **kw):
        calls.sessions.append(1)
        return SimpleNamespace(id=uuid.uuid4())

    async def history(*a, **kw):
        return []

    async def scope(module_id, db):
        return str(module_id), None, None

    monkeypatch.setattr(chat_api, "_get_or_create_session", session)
    monkeypatch.setattr(chat_api, "_load_history", history)
    monkeypatch.setattr(chat_api, "_resolve_module_scope", scope)
    monkeypatch.setattr(chat_api, "get_pipeline", lambda: spy("chat")())

    app.dependency_overrides[require_auth] = lambda: LEARNER
    app.dependency_overrides[get_db] = lambda: FakeDB()
    try:
        yield SimpleNamespace(client=TestClient(app, raise_server_exceptions=False), calls=calls)
    finally:
        app.dependency_overrides.pop(require_auth, None)
        app.dependency_overrides.pop(get_db, None)


GENERATION = [
    ("/api/quiz/generate", {}, "quiz"),
    ("/api/flashcards/generate", {}, "flashcards"),
    ("/api/summarise", {"scope": "module"}, "summary"),
]
S = lambda x: str(x)  # noqa: E731

# (case id, scope fields, expected status). Every one of these must be refused before the AI is reached.
DENIED = [
    ("foreign module", {"module_id": S(FOREIGN_MODULE)}, 403),
    ("nonexistent module", {"module_id": S(UNKNOWN)}, 404),
    ("foreign week alone", {"week_id": S(FOREIGN_WEEK)}, 403),
    ("nonexistent week", {"week_id": S(UNKNOWN)}, 404),
    ("own module + foreign week", {"module_id": S(MY_MODULE), "week_id": S(FOREIGN_WEEK)}, 403),
    ("foreign module + own week", {"module_id": S(FOREIGN_MODULE), "week_id": S(MY_WEEK)}, 403),
    ("document id alone", {"document_id": S(uuid.uuid4())}, 422),
    ("own module + document id", {"module_id": S(MY_MODULE), "document_id": S(uuid.uuid4())}, 422),
    ("foreign module + document id", {"module_id": S(FOREIGN_MODULE), "document_id": S(uuid.uuid4())}, 422),
    ("no scope at all", {}, 422),
]


@pytest.mark.parametrize("path,extra,feature", GENERATION)
@pytest.mark.parametrize("case,scope,expected", DENIED, ids=[d[0] for d in DENIED])
def test_unauthorized_generation_is_refused_before_the_ai_is_reached(api, path, extra, feature, case, scope, expected):
    r = api.client.post(path, json={**extra, **scope})
    assert r.status_code == expected, (case, r.status_code, r.text[:200])
    assert api.calls.ai == [], "the AI provider was reached for an unauthorized request"
    assert api.calls.metered == [], "usage was metered for an unauthorized request"


@pytest.mark.parametrize("path,extra,feature", GENERATION)
@pytest.mark.parametrize("scope", [{"module_id": S(MY_MODULE)}, {"week_id": S(MY_WEEK)},
                                   {"module_id": S(MY_MODULE), "week_id": S(MY_WEEK)}],
                         ids=["own module", "own week", "own module + own week"])
def test_authorized_generation_reaches_the_ai(api, path, extra, feature, scope):
    """Positive control: the guard must not simply block everything."""
    r = api.client.post(path, json={**extra, **scope})
    assert r.status_code == 500 and api.calls.ai == [feature]   # 500 = our spy stopped it at the AI boundary
    assert api.calls.metered == [feature]


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("module_id,expected", [(FOREIGN_MODULE, 403), (UNKNOWN, 404), (None, 422)],
                         ids=["foreign module", "nonexistent module", "no module"])
def test_chat_is_refused_before_session_creation_and_the_ai(api, path, module_id, expected):
    body = {"message": "What is in this module?", "scope_mode": "everything"}
    if module_id is not None:
        body["module_id"] = S(module_id)
    r = api.client.post(path, json=body)
    assert r.status_code == expected, r.text[:200]
    assert api.calls.sessions == [], "a chat session was created for an unauthorized request"
    assert api.calls.ai == [] and api.calls.metered == []


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
def test_authorized_chat_reaches_the_ai(api, path):
    r = api.client.post(path, json={"message": "Explain this", "scope_mode": "everything", "module_id": S(MY_MODULE)})
    assert api.calls.sessions == [1]
    assert api.calls.ai == ["chat"]
    assert r.status_code in (200, 500)   # the spy aborts at the AI boundary; the point is that it was reached


def test_the_five_endpoints_are_the_ones_under_test(api):
    """Guards the guard: if a generation route is renamed or removed this test file must be updated."""
    paths = set(app.openapi()["paths"])
    for p in ("/api/chat", "/api/chat/stream", "/api/quiz/generate", "/api/flashcards/generate", "/api/summarise"):
        assert p in paths
