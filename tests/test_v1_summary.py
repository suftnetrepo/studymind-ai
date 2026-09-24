"""v1 platform summary persistence (POST /summarise saves, GET /summary restores). Real Postgres test DB."""
import itertools

import pytest
from sqlalchemy import func, select

from app.db.models import PlatformSummary
from tests.test_v1_documents import (  # noqa: F401 — pytest fixtures
    api_key, auth, client, db, fakes, pytestmark, session_token, sync_engine, upload,
)


@pytest.fixture
def summaries(monkeypatch):
    counter = itertools.count(1)
    monkeypatch.setattr("app.agents.ai_features.generate_summary",
                        lambda **kw: (f"## Summary {next(counter)} ({kw.get('topic') or 'all'}, {kw['complexity']})", 1))


@pytest.fixture
def course(client, api_key, fakes):
    upload(client, api_key, course_id="c1", user_id="tutor_1")
    return "c1"


def summarise(client, token, **body):
    r = client.post("/api/v1/summarise", headers=auth(token), json=body)
    assert r.status_code == 200, r.text
    return r.json()


def get_summary(client, token, **params):
    r = client.get("/api/v1/summary", headers=auth(token), params=params)
    assert r.status_code == 200, r.text
    return r.json()


class TestSummaryPersistence:
    def test_none_saved(self, client, api_key, course):
        assert get_summary(client, api_key, course_id=course, user_id="stu_1") == {
            "summary": None, "topic": None, "complexity": None, "created_at": None,
        }

    def test_generate_then_restore(self, client, api_key, course, summaries):
        res = summarise(client, api_key, course_id=course, user_id="stu_1", complexity="expert")
        assert res["summary"] == "## Summary 1 (all, expert)" and res["topic"] is None
        saved = get_summary(client, api_key, course_id=course, user_id="stu_1")
        assert saved["summary"] == res["summary"]
        assert saved["complexity"] == "expert" and saved["topic"] is None and saved["created_at"]

    def test_regenerate_same_topic_replaces(self, client, db, api_key, course, summaries):
        summarise(client, api_key, course_id=course, user_id="stu_1")
        summarise(client, api_key, course_id=course, user_id="stu_1", complexity="simple")
        assert db.scalar(select(func.count()).select_from(PlatformSummary)) == 1
        assert get_summary(client, api_key, course_id=course, user_id="stu_1")["summary"] == "## Summary 2 (all, simple)"

    def test_topics_kept_separately_and_latest_wins(self, client, db, api_key, course, summaries):
        summarise(client, api_key, course_id=course, user_id="stu_1")                    # 1: all
        summarise(client, api_key, course_id=course, user_id="stu_1", topic="Loops")     # 2: Loops
        assert db.scalar(select(func.count()).select_from(PlatformSummary)) == 2
        latest = get_summary(client, api_key, course_id=course, user_id="stu_1")
        assert latest["summary"].startswith("## Summary 2") and latest["topic"] == "Loops"
        assert get_summary(client, api_key, course_id=course, user_id="stu_1", topic="")["summary"].startswith("## Summary 1")
        assert get_summary(client, api_key, course_id=course, user_id="stu_1", topic="Loops")["summary"].startswith("## Summary 2")
        assert get_summary(client, api_key, course_id=course, user_id="stu_1", topic="Other")["summary"] is None

    def test_isolated_per_user_and_course(self, client, api_key, course, summaries, fakes):
        upload(client, api_key, course_id="c2", user_id="tutor_1")
        summarise(client, api_key, course_id=course, user_id="stu_1")
        assert get_summary(client, api_key, course_id=course, user_id="stu_2")["summary"] is None
        assert get_summary(client, api_key, course_id="c2", user_id="stu_1")["summary"] is None

    def test_session_token(self, client, api_key, course, summaries):
        token = session_token(client, api_key, course_id=course, user_id="stu_1", role="student")
        summarise(client, token)
        assert get_summary(client, token)["summary"].startswith("## Summary 1")
        other = client.get("/api/v1/summary", headers=auth(token), params={"user_id": "stu_2"})
        assert other.status_code == 401

    def test_requires_auth(self, client):
        assert client.get("/api/v1/summary", params={"course_id": "c1", "user_id": "u"}).status_code == 401
