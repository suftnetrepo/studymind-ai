"""
GET /api/v1/usage — self-service usage stats for the calling platform API key.
Real Postgres test DB (fixtures shared with test_v1_documents); uploads and chat go
through the real endpoints with the ingestor and RAG pipeline faked.
"""
import uuid

from sqlalchemy import select

from app.auth.api_key_auth import generate_api_key
from app.db.models import ApiKey, PlatformCourse
from tests.test_v1_chat_history import chat, pipeline  # noqa: F401 — pytest fixture
from tests.test_v1_documents import (  # noqa: F401 — pytest fixtures
    api_key, auth, client, db, fakes, pytestmark, sync_engine, upload,
)


def usage(client, token, expected=200):
    r = client.get("/api/v1/usage", headers=auth(token))
    assert r.status_code == expected, r.text
    return r.json()


def other_key(db) -> str:
    full_key, key_hash, prefix = generate_api_key()
    db.add(ApiKey(id=uuid.uuid4(), name="Other", key_hash=key_hash, key_prefix=prefix,
                  platform="other", owner_email="o@test.dev", is_active=True, request_count=0))
    db.commit()
    return full_key


class TestUsage:
    def test_returns_usage_for_valid_key(self, client, api_key, fakes, pipeline):
        assert upload(client, api_key, course_id="c1", user_id="tutor_1").status_code in (200, 202)
        # Two students on the same course: one PlatformCourse row each, one shared module
        chat(client, api_key, "What is a variable?", course_id="c1", user_id="stu_1")
        chat(client, api_key, "And a list?",         course_id="c1", user_id="stu_1")
        chat(client, api_key, "What is a loop?",     course_id="c1", user_id="stu_2")

        u = usage(client, api_key)
        assert u["platform"] == "learnify"
        assert u["total_requests"] > 0 and u["last_used_at"]
        assert u["courses_indexed"] == 1
        assert u["documents_uploaded"] == 1
        assert u["total_chunks"] > 0
        # Counted once per session/question — not multiplied by the course's per-user rows
        assert u["total_chat_sessions"] == 2
        assert u["total_questions"] == 3
        assert [c["course_id"] for c in u["courses"]] == ["c1"]
        course = u["courses"][0]
        assert course["status"] == "ready" and course["chunk_count"] > 0 and course["indexed_at"]

    def test_ingested_course_counted_once_across_users(self, client, db, api_key, fakes, pipeline):
        """Course content ingested per user = one PlatformCourse row each, one shared module."""
        body = {"course_id": "c2", "title": "Python 101",
                "sections": [{"title": "Basics", "lectures": [{"title": "Variables"}]}]}
        for user in ("stu_1", "stu_2", "stu_3"):
            r = client.post("/api/v1/courses/ingest", headers=auth(api_key), json={**body, "user_id": user})
            assert r.status_code == 202, r.text
        rows = db.scalars(select(PlatformCourse).where(PlatformCourse.platform_course_id == "c2")).all()
        assert len(rows) == 3 and len({r.module_id for r in rows}) == 1
        for r in rows:  # mark indexed, as the background ingest would
            r.status, r.chunk_count = "ready", 4
        db.commit()

        chat(client, api_key, "Q from stu_1", course_id="c2", user_id="stu_1")
        chat(client, api_key, "Q from stu_2", course_id="c2", user_id="stu_2")

        u = usage(client, api_key)
        assert [(c["course_id"], c["title"], c["status"], c["chunk_count"]) for c in u["courses"]] == [
            ("c2", "Python 101", "ready", 4),
        ]
        assert u["courses_indexed"] == 1
        assert u["documents_uploaded"] == 0
        # 3 rows share the module — sessions/questions must not be tripled
        assert u["total_chat_sessions"] == 2
        assert u["total_questions"] == 2

    def test_filters_out_test_courses(self, client, api_key, fakes, pipeline):
        upload(client, api_key, course_id="c1", user_id="tutor_1")
        upload(client, api_key, course_id="test_course_001", user_id="tutor_1")
        upload(client, api_key, course_id="probe_course", user_id="tutor_1")
        chat(client, api_key, "Test question", course_id="test_course_001", user_id="stu_1")

        u = usage(client, api_key)
        assert [c["course_id"] for c in u["courses"]] == ["c1"]
        assert u["courses_indexed"] == 1
        assert u["documents_uploaded"] == 1
        assert u["total_questions"] == 0

    def test_only_reports_the_callers_own_usage(self, client, db, api_key, fakes, pipeline):
        upload(client, api_key, course_id="c1", user_id="tutor_1")
        chat(client, api_key, "Mine", course_id="c1", user_id="stu_1")
        other = other_key(db)
        upload(client, other, course_id="c9", user_id="tutor_9")

        u = usage(client, other)
        assert u["platform"] == "other"
        assert [c["course_id"] for c in u["courses"]] == ["c9"]
        assert u["total_questions"] == 0

    def test_polling_usage_does_not_inflate_request_count(self, client, db, api_key):
        before = usage(client, api_key)["total_requests"]
        usage(client, api_key)
        usage(client, api_key)
        assert usage(client, api_key)["total_requests"] == before
        key = db.scalars(select(ApiKey)).one()
        db.refresh(key)
        assert key.request_count == before

    def test_empty_key_returns_zeroes(self, client, api_key):
        u = usage(client, api_key)
        assert u["courses"] == [] and u["courses_indexed"] == 0
        assert u["documents_uploaded"] == 0 and u["total_chunks"] == 0
        assert u["total_chat_sessions"] == 0 and u["total_questions"] == 0

    def test_401_for_invalid_key(self, client, api_key):
        usage(client, "sm_live_" + "0" * 64, expected=401)
        usage(client, "not-a-key", expected=401)
        r = client.get("/api/v1/usage")
        assert r.status_code in (401, 403)

    def test_401_for_revoked_key(self, client, db, api_key):
        key = db.scalars(select(ApiKey)).one()
        key.is_active = False
        db.commit()
        usage(client, api_key, expected=401)
