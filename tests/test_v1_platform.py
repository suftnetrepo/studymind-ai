"""
v1 Platform API — API key auth, key admin, course ingestion and AI endpoints.
No live DB, Typesense or LLM: get_db is overridden with an in-memory FakeSession
and the indexing/AI calls are monkeypatched.
"""
import hashlib
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.sql.dml import Update
from sqlalchemy.sql.elements import BindParameter, True_

from app.api.main import app
from app.api.v1 import platform as v1_platform
from app.auth.api_key_auth import generate_api_key, generate_session_token, hash_api_key
from app.auth.dependencies import get_current_user
from app.db.engine import get_db
from app.db.models import ApiKey, Module, PlatformCourse, SessionToken, User


# ── In-memory session ──────────────────────────────────────────────────────

class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    """
    Just enough AsyncSession for these endpoints: equality WHERE clauses on the
    selected entity, plus the request-count UPDATE in get_api_key.
    """
    DEFAULTS = {"is_active": True, "request_count": 0, "chunk_count": 0, "status": "pending"}

    def __init__(self):
        self.objects: list = []
        self.commits = 0

    def add(self, obj):
        for attr, default in self.DEFAULTS.items():
            if hasattr(type(obj), attr) and getattr(obj, attr, None) is None:
                setattr(obj, attr, default)
        if hasattr(type(obj), "created_at") and obj.created_at is None:
            obj.created_at = datetime.now(timezone.utc)
        self.objects.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass

    async def get(self, cls, pk):
        return next((o for o in self.objects if isinstance(o, cls) and o.id == pk), None)

    def of(self, cls):
        return [o for o in self.objects if isinstance(o, cls)]

    async def execute(self, stmt):
        params = stmt.compile().params
        if isinstance(stmt, Update):
            key = await self.get(ApiKey, params["id_1"])
            key.request_count += 1
            key.last_used_at = params["last_used_at"]
            return FakeResult([])

        entity = stmt.column_descriptions[0]["entity"]
        rows   = self.of(entity)
        for crit in stmt._where_criteria:
            if crit.left.table is not entity.__table__:
                continue  # joined-table filters are ignored by the fake
            value = crit.right.value if isinstance(crit.right, BindParameter) else isinstance(crit.right, True_)
            rows  = [r for r in rows if getattr(r, crit.left.key) == value]
        return FakeResult(rows)


def make_user(role: str) -> User:
    return User(id=uuid.uuid4(), email=f"{role}@test.dev", full_name=role, role=role,
                password_hash="x", is_active=True)


def make_key(db: FakeSession, **overrides) -> tuple[str, ApiKey]:
    full_key, key_hash, prefix = generate_api_key()
    key = ApiKey(id=uuid.uuid4(), name="Test", key_hash=key_hash, key_prefix=prefix,
                 platform="learnify", owner_email="owner@test.dev", **overrides)
    db.add(key)
    return full_key, key


def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def db():
    return FakeSession()


@pytest.fixture
def client(db):
    async def _get_db():
        yield db
    app.dependency_overrides[get_db] = _get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def as_user(role: str) -> User:
    user = make_user(role)
    app.dependency_overrides[get_current_user] = lambda: user
    return user


INGEST_BODY = {
    "course_id": "course_001",
    "user_id":   "student_001",
    "title":     "Introduction to Python",
    "description": "Learn Python from scratch",
    "sections": [{
        "title": "Variables",
        "lectures": [{"title": "What is a variable?", "description": "A variable stores data."}],
    }],
}


# ── Key generation / hashing ───────────────────────────────────────────────

class TestKeyGeneration:
    def test_format(self):
        full_key, key_hash, prefix = generate_api_key()
        assert re.fullmatch(r"sm_live_[0-9a-f]{64}", full_key)
        assert prefix == full_key[:12] and prefix.startswith("sm_live_")
        assert key_hash == hashlib.sha256(full_key.encode()).hexdigest()

    def test_keys_are_unique(self):
        assert generate_api_key()[0] != generate_api_key()[0]

    def test_hash_deterministic(self):
        assert hash_api_key("sm_live_abc") == hash_api_key("sm_live_abc")
        assert hash_api_key("sm_live_abc") != hash_api_key("sm_live_abd")


# ── Admin endpoints ────────────────────────────────────────────────────────

class TestAdmin:
    def test_create_key_returns_201_and_full_key_once(self, client, db):
        as_user("admin")
        r = client.post("/api/v1/admin/api-keys", json={
            "name": "Learnify Production", "platform": "learnify", "owner_email": "a@b.com",
        })
        assert r.status_code == 201
        body = r.json()
        assert body["key"].startswith("sm_live_")
        stored = db.of(ApiKey)[0]
        assert stored.key_hash == hash_api_key(body["key"])
        assert body["key"] not in {stored.key_hash, stored.key_prefix}

        listed = client.get("/api/v1/admin/api-keys").json()
        assert "key" not in listed[0] and "key_hash" not in listed[0]
        assert listed[0]["prefix"] == body["prefix"]

    def test_create_key_rejects_bad_platform(self, client):
        as_user("admin")
        r = client.post("/api/v1/admin/api-keys", json={
            "name": "x", "platform": "Bad Platform!", "owner_email": "a@b.com",
        })
        assert r.status_code == 422

    @pytest.mark.parametrize("role", ["student", "lecturer", "self_learner"])
    def test_non_admin_forbidden(self, client, role):
        as_user(role)
        assert client.get("/api/v1/admin/api-keys").status_code == 403
        assert client.post("/api/v1/admin/api-keys", json={
            "name": "x", "platform": "learnify", "owner_email": "a@b.com",
        }).status_code == 403

    def test_list_requires_jwt(self, client):
        assert client.get("/api/v1/admin/api-keys").status_code in (401, 403)

    def test_list_keys(self, client, db):
        as_user("admin")
        make_key(db)
        make_key(db)
        r = client.get("/api/v1/admin/api-keys")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_revoke_sets_inactive(self, client, db):
        as_user("admin")
        _, key = make_key(db)
        r = client.delete(f"/api/v1/admin/api-keys/{key.id}")
        assert r.status_code == 200 and r.json() == {"revoked": True}
        assert key.is_active is False

    def test_revoke_unknown_404_and_bad_id_422(self, client):
        as_user("admin")
        assert client.delete(f"/api/v1/admin/api-keys/{uuid.uuid4()}").status_code == 404
        assert client.delete("/api/v1/admin/api-keys/not-a-uuid").status_code == 422

    def test_usage(self, client, db):
        as_user("admin")
        _, key = make_key(db, request_count=7)
        db.add(PlatformCourse(id=uuid.uuid4(), api_key_id=key.id, platform_course_id="c1",
                              platform_user_id="u1", course_title="Course 1", status="ready"))
        body = client.get(f"/api/v1/admin/api-keys/{key.id}/usage").json()
        assert body["request_count"] == 7
        assert body["courses"][0]["course_id"] == "c1"


# ── API key auth ───────────────────────────────────────────────────────────

class TestApiKeyAuth:
    URL = "/api/v1/courses/status?course_id=c&user_id=u"

    def test_no_key_401(self, client):
        r = client.get(self.URL)
        assert r.status_code == 401
        assert r.json()["detail"] == "Authentication required"

    def test_wrong_key_401(self, client, db):
        make_key(db)
        wrong, _, _ = generate_api_key()
        r = client.get(self.URL, headers=auth(wrong))
        assert r.status_code == 401
        assert r.json()["detail"] == "Invalid API key"

    def test_jwt_style_token_401(self, client):
        assert client.get(self.URL, headers=auth("eyJhbGciOi.jwt.token")).status_code == 401

    def test_revoked_key_401(self, client, db):
        full_key, _ = make_key(db, is_active=False)
        assert client.get(self.URL, headers=auth(full_key)).status_code == 401

    def test_expired_key_401(self, client, db):
        full_key, _ = make_key(db, expires_at=datetime.now(timezone.utc) - timedelta(days=1))
        r = client.get(self.URL, headers=auth(full_key))
        assert r.status_code == 401
        assert r.json()["detail"] == "API key expired"

    def test_valid_key_tracks_usage(self, client, db):
        full_key, key = make_key(db)
        assert client.get(self.URL, headers=auth(full_key)).status_code == 200
        client.get(self.URL, headers=auth(full_key))
        assert key.request_count == 2
        assert key.last_used_at is not None


# ── Course ingestion + status ──────────────────────────────────────────────

class TestIngest:
    @pytest.fixture(autouse=True)
    def no_indexing(self, monkeypatch):
        self.index_calls = []

        async def fake_index(pc_id, text):
            self.index_calls.append((pc_id, text))
        monkeypatch.setattr(v1_platform, "index_course_content", fake_index)

    def test_ingest_creates_module_and_platform_course(self, client, db):
        full_key, key = make_key(db)
        r = client.post("/api/v1/courses/ingest", json=INGEST_BODY, headers=auth(full_key))
        assert r.status_code == 202
        assert r.json()["status"] == "indexing"

        [pc]     = db.of(PlatformCourse)
        [module] = db.of(Module)
        [user]   = db.of(User)
        assert pc.api_key_id == key.id and pc.module_id == module.id
        assert pc.status == "indexing"
        assert str(module.id) == r.json()["module_id"]
        assert module.owner_id == user.id
        assert user.email == "platform_learnify_student_001@studymind.internal"

        # Background indexing was scheduled with the flattened course text
        [(pc_id, text)] = self.index_calls
        assert pc_id == pc.id
        assert "## Variables" in text and "A variable stores data." in text

    def test_reingest_reuses_module(self, client, db):
        full_key, _ = make_key(db)
        first  = client.post("/api/v1/courses/ingest", json=INGEST_BODY, headers=auth(full_key))
        second = client.post("/api/v1/courses/ingest",
                             json={**INGEST_BODY, "title": "Python 101"}, headers=auth(full_key))
        assert first.json()["module_id"] == second.json()["module_id"]
        assert len(db.of(PlatformCourse)) == 1 and len(db.of(Module)) == 1
        assert db.of(Module)[0].title == "Python 101"

    def test_courses_isolated_per_api_key(self, client, db):
        key_a, _ = make_key(db)
        key_b, _ = make_key(db)
        client.post("/api/v1/courses/ingest", json=INGEST_BODY, headers=auth(key_a))
        r = client.get("/api/v1/courses/status",
                       params={"course_id": "course_001", "user_id": "student_001"},
                       headers=auth(key_b))
        assert r.json()["status"] == "not_found"

    @pytest.mark.parametrize("status", ["pending", "indexing", "ready", "failed"])
    def test_status(self, client, db, status):
        full_key, key = make_key(db)
        db.add(PlatformCourse(id=uuid.uuid4(), api_key_id=key.id, platform_course_id="c1",
                              platform_user_id="u1", module_id=uuid.uuid4(),
                              course_title="C", status=status, chunk_count=3))
        r = client.get("/api/v1/courses/status", params={"course_id": "c1", "user_id": "u1"},
                       headers=auth(full_key))
        assert r.status_code == 200
        assert r.json()["status"] == status

    def test_status_not_found(self, client, db):
        full_key, _ = make_key(db)
        r = client.get("/api/v1/courses/status", params={"course_id": "x", "user_id": "y"},
                       headers=auth(full_key))
        assert r.json()["status"] == "not_found"


class TestIndexCourseContent:
    @pytest.fixture
    def setup(self, db, monkeypatch):
        _, key = make_key(db)
        module = Module(id=uuid.uuid4(), title="C", owner_id=uuid.uuid4(), course_code="LEAC1")
        pc = PlatformCourse(id=uuid.uuid4(), api_key_id=key.id, platform_course_id="c1",
                            platform_user_id="u1", module_id=module.id, course_title="C",
                            status="indexing")
        db.add(module)
        db.add(pc)

        @asynccontextmanager
        async def session():
            yield db
        monkeypatch.setattr(v1_platform, "AsyncSessionLocal", session)
        return pc

    def _ingestor(self, monkeypatch, result):
        class FakeIngestor:
            def ingest(self, *args, **kwargs):
                return result
        monkeypatch.setattr(v1_platform, "get_ingestor", lambda: FakeIngestor())

    @pytest.mark.asyncio
    async def test_success_marks_ready(self, setup, monkeypatch):
        self._ingestor(monkeypatch, {
            "status": "indexed", "chunk_count": 1, "error": None,
            "indexed_at": datetime.now(timezone.utc),
            "chunks": [{"typesense_id": "d__0", "chunk_index": 0, "content": "x",
                        "token_count": 1, "chunk_metadata": {}}],
        })
        await v1_platform.index_course_content(setup.id, "Course: C")
        assert setup.status == "ready" and setup.chunk_count == 1 and setup.indexed_at

    @pytest.mark.asyncio
    async def test_failure_marks_failed(self, setup, monkeypatch):
        self._ingestor(monkeypatch, {"status": "failed", "chunk_count": 0, "error": "boom",
                                     "indexed_at": None, "chunks": []})
        await v1_platform.index_course_content(setup.id, "Course: C")
        assert setup.status == "failed" and setup.error_message == "boom"


# ── AI endpoints ───────────────────────────────────────────────────────────

class TestAiEndpoints:
    AI_CALLS = [
        ("/api/v1/chat",                {"message": "What is a variable?"}),
        ("/api/v1/quiz/generate",       {}),
        ("/api/v1/flashcards/generate", {}),
        ("/api/v1/summarise",           {}),
    ]

    @pytest.mark.parametrize("url,extra", AI_CALLS)
    def test_400_when_not_ingested(self, client, db, url, extra):
        full_key, _ = make_key(db)
        r = client.post(url, json={"course_id": "c1", "user_id": "u1", **extra},
                        headers=auth(full_key))
        assert r.status_code == 400
        assert "not_found" in r.json()["detail"]

    @pytest.mark.parametrize("url,extra", AI_CALLS)
    def test_400_when_still_indexing(self, client, db, url, extra):
        full_key, key = make_key(db)
        db.add(PlatformCourse(id=uuid.uuid4(), api_key_id=key.id, platform_course_id="c1",
                              platform_user_id="u1", module_id=uuid.uuid4(),
                              course_title="C", status="indexing"))
        r = client.post(url, json={"course_id": "c1", "user_id": "u1", **extra},
                        headers=auth(full_key))
        assert r.status_code == 400
        assert "indexing" in r.json()["detail"]

    @pytest.mark.parametrize("url,extra", AI_CALLS)
    def test_401_without_key(self, client, url, extra):
        r = client.post(url, json={"course_id": "c1", "user_id": "u1", **extra})
        assert r.status_code == 401

    def test_quiz_when_ready(self, client, db, monkeypatch):
        full_key, key = make_key(db)
        module_id = uuid.uuid4()
        db.add(PlatformCourse(id=uuid.uuid4(), api_key_id=key.id, platform_course_id="c1",
                              platform_user_id="u1", module_id=module_id,
                              course_title="C", status="ready"))
        seen = {}

        def fake_quiz(**kwargs):
            seen.update(kwargs)
            return [{"question": "Q?"}]
        monkeypatch.setattr("app.agents.ai_features.generate_quiz", fake_quiz)

        r = client.post("/api/v1/quiz/generate",
                        json={"course_id": "c1", "user_id": "u1", "question_count": 3},
                        headers=auth(full_key))
        assert r.status_code == 200
        assert r.json()["count"] == 1
        assert seen["module_id"] == str(module_id) and seen["question_count"] == 3


# ── Session tokens ─────────────────────────────────────────────────────────

def make_session(db: FakeSession, key: ApiKey, course_id="c1", user_id="u1",
                 expires_in=timedelta(hours=1)) -> str:
    token, token_hash = generate_session_token()
    db.add(SessionToken(id=uuid.uuid4(), token_hash=token_hash, api_key_id=key.id,
                        course_id=course_id, user_id=user_id, user_role="student",
                        expires_at=datetime.now(timezone.utc) + expires_in))
    return token


def ready_course(db: FakeSession, key: ApiKey, course_id="c1", user_id="u1") -> PlatformCourse:
    pc = PlatformCourse(id=uuid.uuid4(), api_key_id=key.id, platform_course_id=course_id,
                        platform_user_id=user_id, module_id=uuid.uuid4(),
                        course_title="C", status="ready")
    db.add(pc)
    return pc


class TestCreateSession:
    def test_creates_st_token(self, client, db):
        full_key, key = make_key(db)
        r = client.post("/api/v1/auth/session", headers=auth(full_key),
                        json={"course_id": "c1", "user_id": "u1"})
        assert r.status_code == 201
        body = r.json()
        assert re.fullmatch(r"st_[0-9a-f]{64}", body["session_token"])
        assert body["expires_in"] == 3600 and body["course_status"] == "not_found"

        [st] = db.of(SessionToken)
        assert st.token_hash == hash_api_key(body["session_token"])
        assert body["session_token"] not in {st.token_hash}
        assert (st.api_key_id, st.course_id, st.user_id) == (key.id, "c1", "u1")

    def test_links_module_when_already_ingested(self, client, db):
        full_key, key = make_key(db)
        pc = ready_course(db, key)
        body = client.post("/api/v1/auth/session", headers=auth(full_key),
                           json={"course_id": "c1", "user_id": "u1"}).json()
        assert body["course_status"] == "ready"
        assert db.of(SessionToken)[0].module_id == pc.module_id

    @pytest.mark.parametrize("expires_in", [60, 86401])
    def test_expiry_bounds(self, client, db, expires_in):
        full_key, _ = make_key(db)
        r = client.post("/api/v1/auth/session", headers=auth(full_key),
                        json={"course_id": "c1", "user_id": "u1", "expires_in": expires_in})
        assert r.status_code == 422

    def test_requires_api_key(self, client):
        assert client.post("/api/v1/auth/session",
                           json={"course_id": "c1", "user_id": "u1"}).status_code == 401

    def test_session_token_cannot_mint_tokens(self, client, db):
        _, key = make_key(db)
        token = make_session(db, key)
        r = client.post("/api/v1/auth/session", headers=auth(token),
                        json={"course_id": "other", "user_id": "u1"})
        assert r.status_code == 401


class TestSessionTokenAuth:
    def test_chat_with_session_token(self, client, db, monkeypatch):
        _, key = make_key(db)
        pc = ready_course(db, key)
        token = make_session(db, key)
        seen = {}

        class FakePipeline:
            def query(self, **kwargs):
                seen.update(kwargs)
                return {"answer": "A variable stores data.", "sources": [], "no_content_found": False}
        monkeypatch.setattr("app.agents.rag_pipeline.get_pipeline", lambda: FakePipeline())

        # No course_id/user_id in the body — taken from the token
        r = client.post("/api/v1/chat", headers=auth(token), json={"message": "What is a variable?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "A variable stores data."
        assert seen["module_id"] == str(pc.module_id)

    def test_quiz_with_session_token(self, client, db, monkeypatch):
        _, key = make_key(db)
        pc = ready_course(db, key)
        token = make_session(db, key)
        monkeypatch.setattr("app.agents.ai_features.generate_quiz",
                            lambda **kw: [{"question": "Q?", "module": kw["module_id"]}])
        r = client.post("/api/v1/quiz/generate", headers=auth(token),
                        json={"course_id": "c1", "user_id": "u1", "question_count": 2})
        assert r.status_code == 200
        assert r.json()["questions"][0]["module"] == str(pc.module_id)
        assert r.json()["course_id"] == "c1"

    def test_status_and_usage_tracked(self, client, db):
        _, key = make_key(db)
        ready_course(db, key)
        token = make_session(db, key)
        r = client.get("/api/v1/courses/status", headers=auth(token))
        assert r.json()["status"] == "ready"
        assert key.request_count == 1

    def test_ingest_scoped_to_token_course(self, client, db, monkeypatch):
        async def fake_index(pc_id, text):
            pass
        monkeypatch.setattr(v1_platform, "index_course_content", fake_index)
        _, key = make_key(db)
        token = make_session(db, key, course_id="course_001", user_id="student_001")
        r = client.post("/api/v1/courses/ingest", headers=auth(token),
                        json={"title": "Intro", "sections": []})
        assert r.status_code == 202
        [pc] = db.of(PlatformCourse)
        assert (pc.platform_course_id, pc.platform_user_id) == ("course_001", "student_001")

    def test_expired_token_401(self, client, db):
        _, key = make_key(db)
        ready_course(db, key)
        token = make_session(db, key, expires_in=timedelta(seconds=-1))
        r = client.post("/api/v1/chat", headers=auth(token), json={"message": "hi"})
        assert r.status_code == 401
        assert r.json()["detail"] == "Session token expired"

    def test_unknown_token_401(self, client, db):
        make_key(db)
        fake, _ = generate_session_token()
        r = client.post("/api/v1/chat", headers=auth(fake), json={"message": "hi"})
        assert r.status_code == 401
        assert r.json()["detail"] == "Invalid session token"

    @pytest.mark.parametrize("url,extra", TestAiEndpoints.AI_CALLS)
    def test_wrong_course_401(self, client, db, url, extra):
        _, key = make_key(db)
        ready_course(db, key, course_id="c1")
        ready_course(db, key, course_id="c2")
        token = make_session(db, key, course_id="c1")
        r = client.post(url, headers=auth(token),
                        json={"course_id": "c2", "user_id": "u1", **extra})
        assert r.status_code == 401

    def test_wrong_user_401(self, client, db):
        _, key = make_key(db)
        token = make_session(db, key, user_id="u1")
        r = client.get("/api/v1/courses/status", headers=auth(token),
                       params={"course_id": "c1", "user_id": "someone_else"})
        assert r.status_code == 401

    def test_revoked_api_key_kills_session(self, client, db):
        _, key = make_key(db)
        ready_course(db, key)
        token = make_session(db, key)
        key.is_active = False
        r = client.get("/api/v1/courses/status", headers=auth(token))
        assert r.status_code == 401

    def test_admin_rejects_session_token(self, client, db):
        _, key = make_key(db)
        token = make_session(db, key)
        assert client.get("/api/v1/admin/api-keys", headers=auth(token)).status_code == 401


class TestApiKeyBackwardCompat:
    """sm_live_ keys keep working on every endpoint with course_id/user_id in the request."""

    @pytest.fixture
    def setup(self, db, monkeypatch):
        full_key, key = make_key(db)
        ready_course(db, key)

        class FakePipeline:
            def query(self, **kw):
                return {"answer": "ok", "sources": [], "no_content_found": False}
        monkeypatch.setattr("app.agents.rag_pipeline.get_pipeline", lambda: FakePipeline())
        monkeypatch.setattr("app.agents.ai_features.generate_quiz", lambda **kw: [{"q": 1}])
        monkeypatch.setattr("app.agents.ai_features.generate_flashcards", lambda **kw: [{"c": 1}])
        monkeypatch.setattr("app.agents.ai_features.generate_summary", lambda **kw: ("## S", 1))

        async def fake_index(pc_id, text):
            pass
        monkeypatch.setattr(v1_platform, "index_course_content", fake_index)
        return full_key

    @pytest.mark.parametrize("url,extra", TestAiEndpoints.AI_CALLS)
    def test_ai_endpoints(self, client, setup, url, extra):
        r = client.post(url, headers=auth(setup), json={"course_id": "c1", "user_id": "u1", **extra})
        assert r.status_code == 200

    def test_status_and_ingest(self, client, setup):
        r = client.get("/api/v1/courses/status", headers=auth(setup),
                       params={"course_id": "c1", "user_id": "u1"})
        assert r.json()["status"] == "ready"
        r = client.post("/api/v1/courses/ingest", headers=auth(setup),
                        json={"course_id": "c9", "user_id": "u1", "title": "New"})
        assert r.status_code == 202

    @pytest.mark.parametrize("url,extra", TestAiEndpoints.AI_CALLS)
    def test_api_key_requires_course_and_user(self, client, setup, url, extra):
        r = client.post(url, headers=auth(setup), json=extra)
        assert r.status_code == 422
