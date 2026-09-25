"""
v1 platform chat persistence + /chat/history. Real Postgres test DB (fixtures shared with
test_v1_documents); the RAG pipeline is faked and records the history it receives.
"""
import uuid

import pytest
from sqlalchemy import func, select

from app.api.v1 import platform as v1_platform
from app.db.models import ChatMessage, ChatSession, User
from tests.test_v1_documents import (  # noqa: F401 — pytest fixtures
    api_key, auth, client, db, fakes, pytestmark, session_token, sync_engine, upload,
)


@pytest.fixture
def pipeline(monkeypatch):
    calls = []

    class Pipeline:
        def query(self, **kw):
            calls.append(kw)
            return {"answer": f"Answer {len(calls)}", "sources": [], "no_content_found": False,
                    "latency_ms": 5, "token_count": 10}

    monkeypatch.setattr("app.agents.rag_pipeline.get_pipeline", lambda: Pipeline())
    return calls


@pytest.fixture
def course(client, api_key, fakes):
    """A course that's ready for AI (one indexed upload)."""
    upload(client, api_key, course_id="c1", user_id="tutor_1")
    return "c1"


def chat(client, token, message, **body):
    r = client.post("/api/v1/chat", headers=auth(token), json={"message": message, **body})
    assert r.status_code == 200, r.text
    return r.json()


def history(client, token, **params):
    r = client.get("/api/v1/chat/history", headers=auth(token), params=params)
    assert r.status_code == 200, r.text
    return r.json()


class TestChatPersistence:
    def test_chat_returns_session_and_history_restores_it(self, client, api_key, course, pipeline):
        res = chat(client, api_key, "What is a variable?", course_id=course, user_id="stu_1")
        assert res["session_id"]

        h = history(client, api_key, course_id=course, user_id="stu_1")
        assert h["session_id"] == res["session_id"]
        assert [(m["role"], m["content"]) for m in h["messages"]] == [
            ("user", "What is a variable?"), ("assistant", "Answer 1"),
        ]
        assert all(m["timestamp"] and m["id"] for m in h["messages"])

    def test_follow_up_reuses_session_and_sends_context(self, client, db, api_key, course, pipeline):
        first  = chat(client, api_key, "What is a list?", course_id=course, user_id="stu_1")
        second = chat(client, api_key, "Give an example", course_id=course, user_id="stu_1")
        assert first["session_id"] == second["session_id"]
        assert pipeline[0]["history"] == []
        assert pipeline[1]["history"] == [
            {"role": "user", "content": "What is a list?"},
            {"role": "assistant", "content": "Answer 1"},
        ]
        assert db.scalar(select(func.count()).select_from(ChatSession)) == 1
        assert db.scalar(select(func.count()).select_from(ChatMessage)) == 4

    def test_context_is_the_most_recent_turns(self, client, api_key, course, pipeline, monkeypatch):
        monkeypatch.setattr(v1_platform, "CHAT_HISTORY_TURNS", 1)
        for q in ("Q1", "Q2", "Q3"):
            chat(client, api_key, q, course_id=course, user_id="stu_1")
        assert pipeline[2]["history"] == [
            {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "Answer 2"},
        ]

    def test_session_owned_by_platform_user_with_metadata(self, client, db, api_key, course, pipeline):
        chat(client, api_key, "Hi", course_id=course, user_id="stu_1")
        s = db.scalars(select(ChatSession)).one()
        owner = db.get(User, s.user_id)
        assert owner.email == "platform_learnify_stu_1@studymind.internal"
        assert s.session_metadata["platform_course_id"] == course
        assert s.session_metadata["platform_user_id"] == "stu_1"

    def test_history_isolated_per_user_and_course(self, client, api_key, course, pipeline, fakes):
        upload(client, api_key, course_id="c2", user_id="tutor_1")
        chat(client, api_key, "Mine", course_id=course, user_id="stu_1")
        chat(client, api_key, "Other course", course_id="c2", user_id="stu_1")
        assert history(client, api_key, course_id=course, user_id="stu_2") == {"session_id": None, "messages": []}
        contents = [m["content"] for m in history(client, api_key, course_id=course, user_id="stu_1")["messages"]]
        assert "Mine" in contents and "Other course" not in contents

    def test_empty_history_for_new_user(self, client, api_key, course):
        assert history(client, api_key, course_id=course, user_id="nobody") == {"session_id": None, "messages": []}

    def test_history_limit_returns_latest(self, client, api_key, course, pipeline):
        for q in ("Q1", "Q2", "Q3"):
            chat(client, api_key, q, course_id=course, user_id="stu_1")
        msgs = history(client, api_key, course_id=course, user_id="stu_1", limit=2)["messages"]
        assert [m["content"] for m in msgs] == ["Q3", "Answer 3"]

    def test_history_with_session_token(self, client, api_key, course, pipeline):
        token = session_token(client, api_key, course_id=course, user_id="stu_1", role="student")
        chat(client, token, "From the browser")
        h = history(client, token)  # course/user come from the token
        assert [m["content"] for m in h["messages"]] == ["From the browser", "Answer 1"]

    def test_session_token_cannot_read_another_users_history(self, client, api_key, course, pipeline):
        chat(client, api_key, "Private", course_id=course, user_id="stu_2")
        token = session_token(client, api_key, course_id=course, user_id="stu_1", role="student")
        r = client.get("/api/v1/chat/history", headers=auth(token), params={"user_id": "stu_2"})
        assert r.status_code == 401

    def test_history_requires_auth(self, client):
        assert client.get("/api/v1/chat/history", params={"course_id": "c1", "user_id": "u"}).status_code == 401


class TestNewChat:
    def new_chat(self, client, token, **body):
        r = client.post("/api/v1/chat/new", headers=auth(token), json=body)
        assert r.status_code == 200, r.text
        return r.json()

    def test_new_chat_starts_fresh_session(self, client, db, api_key, course, pipeline):
        first = chat(client, api_key, "Old question", course_id=course, user_id="stu_1")
        assert self.new_chat(client, api_key, course_id=course, user_id="stu_1")["closed_sessions"] == 1
        assert history(client, api_key, course_id=course, user_id="stu_1") == {"session_id": None, "messages": []}

        second = chat(client, api_key, "New question", course_id=course, user_id="stu_1")
        assert second["session_id"] != first["session_id"]
        assert pipeline[-1]["history"] == []  # no context carried over
        assert [m["content"] for m in history(client, api_key, course_id=course, user_id="stu_1")["messages"]] == [
            "New question", "Answer 2",
        ]
        # Old conversation is closed, not deleted
        old = db.get(ChatSession, uuid.UUID(first["session_id"]))
        assert old.is_active is False
        assert db.scalar(select(func.count()).select_from(ChatMessage)) == 4

    def test_new_chat_without_history_is_ok(self, client, api_key, course):
        assert self.new_chat(client, api_key, course_id=course, user_id="nobody")["closed_sessions"] == 0

    def test_new_chat_only_affects_that_user(self, client, api_key, course, pipeline):
        chat(client, api_key, "Mine", course_id=course, user_id="stu_1")
        chat(client, api_key, "Theirs", course_id=course, user_id="stu_2")
        self.new_chat(client, api_key, course_id=course, user_id="stu_1")
        assert len(history(client, api_key, course_id=course, user_id="stu_2")["messages"]) == 2

    def test_new_chat_with_session_token(self, client, api_key, course, pipeline):
        token = session_token(client, api_key, course_id=course, user_id="stu_1", role="student")
        chat(client, token, "Hello")
        assert self.new_chat(client, token)["closed_sessions"] == 1
        assert history(client, token)["messages"] == []
        r = client.post("/api/v1/chat/new", headers=auth(token), json={"user_id": "stu_2"})
        assert r.status_code == 401

    def test_requires_auth(self, client):
        assert client.post("/api/v1/chat/new", json={"course_id": "c1", "user_id": "u"}).status_code == 401
