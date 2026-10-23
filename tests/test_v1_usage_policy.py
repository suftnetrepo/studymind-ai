"""
Usage policies (Learnify free-course AI Tutor Pass): opt-in quotas + cost budget.
Real Postgres test DB; AI calls are faked but report realistic OpenAI usage so cost accounting runs.
"""
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.agents import llm_usage
from app.auth.api_key_auth import generate_api_key
from app.config import get_settings
from app.db.models import ApiKey, SessionToken, SessionTokenPolicy, UsageCounter, UsageEvent, UsagePolicy
from app.usage import policy as usage
from tests.test_v1_documents import (  # noqa: F401 — pytest fixtures
    api_key, auth, client, db, fakes, pytestmark, session_token, sync_engine, upload,
)

settings = get_settings()
FAKE_USAGE = {"usage": {"prompt_tokens": 5000, "completion_tokens": 500}, "model": "gpt-4.1-mini-2025-04-14"}
CHAT_COST = 5000 * 0.40 + 500 * 1.60 + usage.RETRIEVAL_ALLOWANCE_MICRO_USD     # micro-USD per faked call


def _report():
    llm_usage.report(SimpleNamespace(raw=FAKE_USAGE))


# ── Fixtures / helpers ─────────────────────────────────────────────────────

@pytest.fixture
def ai(monkeypatch):
    """Fakes for the four AI features. Each reports OpenAI usage like the real LLM call does."""
    state = SimpleNamespace(calls=[], fail=None)

    def maybe_fail(where):
        if state.fail == where:
            raise RuntimeError("model unavailable")
        if state.fail == f"{where}-after-llm":
            _report()
            raise RuntimeError("failed after the model ran")

    class Pipeline:
        def query(self, **kw):
            maybe_fail("chat")
            state.calls.append("chat")
            _report()
            return {"answer": f"Answer {len(state.calls)}", "sources": [], "no_content_found": False,
                    "latency_ms": 5, "token_count": 5500}

    def quiz(**kw):
        if state.fail == "quiz-novalue":
            raise ValueError("No content found")
        maybe_fail("quiz"); state.calls.append("quiz"); _report()
        return [{"question": "Q", "options": [{"id": "a", "text": "A"}], "correct_answer": "a"}]

    def flashcards(**kw):
        maybe_fail("flashcards"); state.calls.append("flashcards"); _report()
        return [{"front": "F", "back": "B"}]

    def summary(**kw):
        maybe_fail("summary"); state.calls.append("summary"); _report()
        return ("## Summary", 1)

    monkeypatch.setattr("app.agents.rag_pipeline.get_pipeline", lambda: Pipeline())
    monkeypatch.setattr("app.agents.ai_features.generate_quiz", quiz)
    monkeypatch.setattr("app.agents.ai_features.generate_flashcards", flashcards)
    monkeypatch.setattr("app.agents.ai_features.generate_summary", summary)
    return state


@pytest.fixture
def courses(client, api_key, fakes):
    for c in ("js", "py", "react"):
        upload(client, api_key, course_id=c, user_id="tutor_1")
    return ("js", "py", "react")


@pytest.fixture
def limits(monkeypatch):
    """Set pass limits for policies created after this point (they're frozen onto each policy)."""
    def _set(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(settings, k, v)
    return _set


def make_key(db, platform="learnify") -> str:
    full_key, key_hash, prefix = generate_api_key()
    db.add(ApiKey(id=uuid.uuid4(), name=platform, key_hash=key_hash, key_prefix=prefix, platform=platform,
                  owner_email="t@test.dev", is_active=True, request_count=0))
    db.commit()
    return full_key


_PASS_EXPIRY: dict[str, datetime] = {}


@pytest.fixture(autouse=True)
def _fixed_pass_expiry():
    """Like Learnify, send each pass's stored, fixed expiry on every token request."""
    _PASS_EXPIRY.clear()
    yield


def mint(client, key, course_id="js", user_id="stu_1", role="student", policy="default", expires_in=3600):
    body = {"course_id": course_id, "user_id": user_id, "user_role": role, "expires_in": expires_in}
    if policy == "default":
        expiry = _PASS_EXPIRY.setdefault("pass-1", usage._now() + timedelta(days=30))
        policy = {"type": usage.PASS_TYPE, "pass_id": "pass-1", "expires_at": expiry.isoformat()}
    if policy is not None:
        body["usage_policy"] = policy
    return client.post("/api/v1/auth/session", headers=auth(key), json=body)


def pass_token(client, key, **kw) -> str:
    r = mint(client, key, **kw)
    assert r.status_code == 201, r.text
    return r.json()["session_token"]


def ask(client, token, message="Explain closures"):
    return client.post("/api/v1/chat", headers=auth(token), json={"message": message})


def gen(client, token, feature, **body):
    path = {"quiz": "/api/v1/quiz/generate", "flashcards": "/api/v1/flashcards/generate",
            "summary": "/api/v1/summarise"}[feature]
    return client.post(path, headers=auth(token), json=body)


def status(client, key, pass_id="pass-1", user_id="stu_1"):
    r = client.get("/api/v1/usage/policy", headers=auth(key), params={"pass_id": pass_id, "user_id": user_id})
    assert r.status_code == 200, r.text
    return r.json()


def policy_row(db, pass_id="pass-1") -> UsagePolicy:
    db.expire_all()
    return db.scalars(select(UsagePolicy).where(UsagePolicy.external_ref == pass_id)).one()


# ── 1–3. Sessions without a policy are untouched ───────────────────────────

class TestUnrestrictedSessions:
    def test_legacy_student_session_is_unlimited(self, client, db, api_key, courses, ai, limits):
        limits(pass_daily_chat=2)
        token = session_token(client, api_key, course_id="js", user_id="stu_1", role="student")
        for _ in range(5):
            assert ask(client, token).status_code == 200
        assert db.scalar(select(func.count()).select_from(UsagePolicy)) == 0
        assert db.scalar(select(func.count()).select_from(UsageCounter)) == 0
        assert db.scalar(select(func.count()).select_from(UsageEvent)) == 0

    def test_legacy_session_response_shape_unchanged(self, client, api_key, courses):
        r = mint(client, api_key, policy=None)
        assert r.status_code == 201
        assert set(r.json()) == {"session_token", "expires_in", "expires_at", "course_id", "user_id", "course_status"}
        assert r.json()["expires_in"] == 3600

    def test_api_key_calls_are_unlimited(self, client, api_key, courses, ai, limits):
        limits(pass_daily_chat=1)
        pass_token(client, api_key)   # a pass exists for this student…
        for _ in range(3):           # …but server-side API-key calls are never limited
            r = client.post("/api/v1/chat", headers=auth(api_key),
                            json={"message": "hi", "course_id": "js", "user_id": "stu_1"})
            assert r.status_code == 200

    def test_other_applications_unaffected_and_cannot_create_policies(self, client, db, courses, ai, limits):
        limits(pass_daily_chat=1)
        other = make_key(db, platform="revvo-mobile")
        upload(client, other, course_id="bio", user_id="t1")
        token = session_token(client, other, course_id="bio", user_id="u1", role="student")
        for _ in range(3):
            assert ask(client, token).status_code == 200
        r = mint(client, other, course_id="bio", user_id="u1")
        assert r.status_code == 403 and r.json()["code"] == "policy_not_permitted"
        assert db.scalar(select(func.count()).select_from(UsagePolicy)) == 0

    def test_paid_course_session_unlimited_while_pass_session_limited(self, client, api_key, courses, ai, limits):
        limits(pass_daily_chat=1)
        free = pass_token(client, api_key, course_id="js")
        paid = session_token(client, api_key, course_id="react", user_id="stu_1", role="student")  # no policy
        assert ask(client, free).status_code == 200
        assert ask(client, free).status_code == 429
        for _ in range(3):
            assert ask(client, paid).status_code == 200


# ── 4–5. Pass sessions are limited, and share one allowance ────────────────

class TestPassLimits:
    def test_daily_chat_limit(self, client, api_key, courses, ai, limits):
        limits(pass_daily_chat=3)
        token = pass_token(client, api_key)
        for _ in range(3):
            assert ask(client, token).status_code == 200
        r = ask(client, token)
        body = r.json()
        assert r.status_code == 429 and body["code"] == "daily_limit_reached"
        assert isinstance(body["detail"], str) and "3 tutor questions" in body["detail"]   # SDK shows detail
        assert body["feature"] == "chat" and body["limit"] == 3 and body["resets_at"]
        assert int(r.headers["Retry-After"]) > 0
        assert ai.calls.count("chat") == 3                       # the 4th never reached the model

    def test_pass_session_response_includes_status(self, client, api_key, courses):
        r = mint(client, api_key)
        assert r.status_code == 201
        up = r.json()["usage_policy"]
        assert up["pass_id"] == "pass-1" and up["status"] == "active" and up["timezone"] == "Europe/London"
        assert up["features"]["chat"] == {"limit": 60, "used": 0, "remaining": 60, "available": True}
        assert "budget" not in up                                # no cost data to the browser side

    def test_default_allowances(self, client, db, api_key, courses):
        pass_token(client, api_key)
        p = policy_row(db)
        assert p.daily_limits == {"chat": 60, "quiz": 5, "flashcards": 5, "summary": 3}
        assert p.budget_micro_usd == 4_500_000 and p.timezone == "Europe/London"

    def test_generation_limits(self, client, api_key, courses, ai, limits):
        limits(pass_daily_quiz=2, pass_daily_flashcards=1, pass_daily_summary=1)
        token = pass_token(client, api_key)
        assert gen(client, token, "quiz").status_code == 200
        assert gen(client, token, "quiz").status_code == 200
        assert gen(client, token, "quiz").json()["code"] == "daily_limit_reached"
        assert gen(client, token, "flashcards").status_code == 200
        assert gen(client, token, "flashcards").status_code == 429
        assert gen(client, token, "summary").status_code == 200
        assert gen(client, token, "summary").status_code == 429
        assert ask(client, token).status_code == 200             # features are counted separately

    def test_allowance_shared_across_courses_tokens_and_tabs(self, client, api_key, courses, ai, limits):
        limits(pass_daily_chat=6)
        js_a, js_b = pass_token(client, api_key, course_id="js"), pass_token(client, api_key, course_id="js")
        py, react  = pass_token(client, api_key, course_id="py"), pass_token(client, api_key, course_id="react")
        for token in (js_a, js_b, py, py, react):                # two "tabs" on js, then other courses
            assert ask(client, token).status_code == 200
        assert status(client, api_key)["features"]["chat"] == {"limit": 6, "used": 5, "remaining": 1, "available": True}
        assert ask(client, pass_token(client, api_key, course_id="py")).status_code == 200   # refreshed token
        assert ask(client, js_a).status_code == 429

    def test_status_endpoint_for_session_token_hides_budget(self, client, api_key, courses, ai):
        token = pass_token(client, api_key)
        ask(client, token)
        r = client.get("/api/v1/usage/policy", headers=auth(token))
        assert r.status_code == 200
        assert r.json()["features"]["chat"]["used"] == 1 and "budget" not in r.json()
        assert "budget" in status(client, api_key)               # server-to-server includes cost data

    def test_topic_cap_applies_to_pass_sessions_only(self, client, api_key, courses, ai):
        long_topic = "x" * (usage.MAX_TOPIC_CHARS + 1)
        r = gen(client, pass_token(client, api_key), "quiz", topic=long_topic)
        assert r.status_code == 422 and r.json()["code"] == "topic_too_long"
        legacy = session_token(client, api_key, course_id="js", user_id="stu_2", role="student")
        assert gen(client, legacy, "quiz", topic=long_topic).status_code == 200


# ── 6–7. Expired, revoked and invalid policies fail closed ─────────────────

class TestPolicyLifecycle:
    def test_expired_policy_rejected(self, client, api_key, courses, ai, monkeypatch):
        token = pass_token(client, api_key)
        assert ask(client, token).status_code == 200
        later = usage._now() + timedelta(days=31)
        monkeypatch.setattr(usage, "_now", lambda: later)
        r = ask(client, token)
        assert r.status_code == 403 and r.json()["code"] == "policy_expired"
        assert client.get("/api/v1/chat/history", headers=auth(token)).status_code == 403   # token unusable
        r = mint(client, api_key, policy={"type": usage.PASS_TYPE, "pass_id": "pass-1",
                                          "expires_at": (later - timedelta(days=1)).isoformat()})
        assert r.status_code == 403 and r.json()["code"] == "policy_expired"

    def test_revoked_policy_rejected_and_cannot_be_revived(self, client, db, api_key, courses, ai):
        token = pass_token(client, api_key)
        expires = policy_row(db).expires_at.isoformat()
        r = client.post("/api/v1/usage/policy/revoke", headers=auth(api_key),
                        json={"pass_id": "pass-1", "user_id": "stu_1", "reason": "refund"})
        assert r.status_code == 200 and r.json()["status"] == "revoked"
        assert ask(client, token).json()["code"] == "policy_revoked"
        again = client.post("/api/v1/usage/policy/revoke", headers=auth(api_key),
                            json={"pass_id": "pass-1", "user_id": "stu_1"})
        assert again.status_code == 200                           # idempotent
        replay = mint(client, api_key, policy={"type": usage.PASS_TYPE, "pass_id": "pass-1", "expires_at": expires})
        assert replay.status_code == 403 and replay.json()["code"] == "policy_revoked"
        assert policy_row(db).revoke_reason == "refund"

    def test_revoke_is_server_to_server_only(self, client, api_key, courses):
        token = pass_token(client, api_key)
        r = client.post("/api/v1/usage/policy/revoke", headers=auth(token), json={"pass_id": "pass-1", "user_id": "stu_1"})
        assert r.status_code == 403

    @pytest.mark.parametrize("policy,status_code,code", [
        ({"type": usage.PASS_TYPE, "pass_id": "p", "expires_at": "PLUS30", "daily_limits": {"chat": 9999}}, 422, None),
        ({"type": usage.PASS_TYPE, "pass_id": "p", "expires_at": "PLUS30", "budget_usd": 999}, 422, None),
        ({"type": "unlimited", "pass_id": "p", "expires_at": "PLUS30"}, 422, None),
        ({"type": usage.PASS_TYPE, "pass_id": "", "expires_at": "PLUS30"}, 422, None),
        ({"type": usage.PASS_TYPE, "pass_id": "p", "expires_at": "PLUS90"}, 422, "policy_invalid"),
        ({"type": usage.PASS_TYPE, "pass_id": "p", "expires_at": "NAIVE"}, 422, "policy_invalid"),
        ({"type": usage.PASS_TYPE, "pass_id": "p", "expires_at": "PAST"}, 403, "policy_expired"),
    ])
    def test_invalid_policies_never_mint_a_token(self, client, db, api_key, courses, policy, status_code, code):
        now = usage._now()
        values = {"PLUS30": now + timedelta(days=30), "PLUS90": now + timedelta(days=90),
                  "PAST": now - timedelta(hours=1)}
        policy = dict(policy)
        if policy["expires_at"] == "NAIVE":
            policy["expires_at"] = (now + timedelta(days=5)).replace(tzinfo=None).isoformat()
        else:
            policy["expires_at"] = values[policy["expires_at"]].isoformat()
        r = mint(client, api_key, policy=policy)
        assert r.status_code == status_code, r.text
        if code:
            assert r.json()["code"] == code
        assert db.scalar(select(func.count()).select_from(SessionToken)) == 0

    def test_policy_only_for_student_sessions(self, client, api_key, courses):
        r = mint(client, api_key, user_id="tutor_1", role="tutor")
        assert r.status_code == 422 and r.json()["code"] == "policy_invalid"

    def test_policy_immutable_no_replacement(self, client, db, api_key, courses):
        pass_token(client, api_key)
        p = policy_row(db)
        other_expiry = (p.expires_at - timedelta(days=3)).isoformat()
        r = mint(client, api_key, policy={"type": usage.PASS_TYPE, "pass_id": "pass-1", "expires_at": other_expiry})
        assert r.status_code == 409 and r.json()["code"] == "policy_conflict"
        r = mint(client, api_key, policy={"type": usage.PASS_TYPE, "pass_id": "pass-1",
                                          "expires_at": p.expires_at.isoformat()})
        assert r.status_code == 201                              # identical re-attach is fine
        assert db.scalar(select(func.count()).select_from(UsagePolicy)) == 1

    def test_tampered_policy_link_fails_closed(self, client, db, api_key, courses, ai):
        token = pass_token(client, api_key)
        p = policy_row(db)
        p.platform_user_id = "someone_else"
        db.commit()
        r = ask(client, token)
        assert r.status_code == 403 and r.json()["code"] == "policy_invalid"

    def test_delisted_platform_fails_closed(self, client, api_key, courses, ai, monkeypatch):
        token = pass_token(client, api_key)
        monkeypatch.setattr(settings, "usage_policy_platforms", "someone-else")
        r = ask(client, token)
        assert r.status_code == 403 and r.json()["code"] == "policy_invalid"


# ── 12. Bound to one student and one integration ───────────────────────────

class TestBinding:
    def test_pass_cannot_be_used_by_another_student(self, client, api_key, courses):
        pass_token(client, api_key, user_id="stu_1")
        r = mint(client, api_key, user_id="stu_2")               # same pass id, different student
        assert r.status_code == 409 and r.json()["code"] == "policy_conflict"

    def test_token_scoped_to_its_student(self, client, api_key, courses, ai):
        token = pass_token(client, api_key)
        r = client.post("/api/v1/chat", headers=auth(token), json={"message": "hi", "user_id": "stu_2"})
        assert r.status_code == 401

    def test_other_integration_cannot_read_or_revoke(self, client, db, api_key, courses):
        pass_token(client, api_key)
        second = make_key(db, platform="learnify")              # a different key, even on the same platform
        r = client.get("/api/v1/usage/policy", headers=auth(second), params={"pass_id": "pass-1", "user_id": "stu_1"})
        assert r.status_code == 404
        r = client.post("/api/v1/usage/policy/revoke", headers=auth(second), json={"pass_id": "pass-1", "user_id": "stu_1"})
        assert r.status_code == 404
        assert status(client, api_key)["status"] == "active"

    def test_status_does_not_reveal_other_students(self, client, api_key, courses):
        pass_token(client, api_key, user_id="stu_1")
        r = client.get("/api/v1/usage/policy", headers=auth(api_key), params={"pass_id": "pass-1", "user_id": "stu_2"})
        assert r.status_code == 404 and r.json()["code"] == "policy_not_found"


# ── 9–10. Failures release; the budget is enforced ─────────────────────────

class TestFailuresAndBudget:
    def test_failed_request_releases_reservation(self, client, db, api_key, courses, ai):
        token = pass_token(client, api_key)
        ai.fail = "chat"
        with pytest.raises(RuntimeError):
            ask(client, token)
        p = policy_row(db)
        counter = db.scalars(select(UsageCounter).where(UsageCounter.policy_id == p.id)).one()
        assert counter.used == 0 and p.reserved_micro_usd == 0 and p.spent_micro_usd == 0
        assert db.scalars(select(UsageEvent)).one().status == "released"

    def test_value_error_422_releases(self, client, db, api_key, courses, ai):
        token = pass_token(client, api_key)
        ai.fail = "quiz-novalue"
        assert gen(client, token, "quiz").status_code == 422
        assert status(client, api_key)["features"]["quiz"]["used"] == 0

    def test_cost_incurred_before_a_failure_is_still_charged(self, client, db, api_key, courses, ai):
        token = pass_token(client, api_key)
        ai.fail = "chat-after-llm"
        with pytest.raises(RuntimeError):
            ask(client, token)
        p = policy_row(db)
        assert status(client, api_key)["features"]["chat"]["used"] == 0      # allowance given back…
        assert p.spent_micro_usd == 5000 * 0.40 + 500 * 1.60 and p.reserved_micro_usd == 0   # …real spend kept

    def test_success_charges_measured_cost(self, client, db, api_key, courses, ai):
        ask(client, pass_token(client, api_key))
        p = policy_row(db)
        e = db.scalars(select(UsageEvent)).one()
        assert p.spent_micro_usd == CHAT_COST and p.reserved_micro_usd == 0
        assert (e.status, e.input_tokens, e.output_tokens, e.model) == ("committed", 5000, 500, "gpt-4.1-mini-2025-04-14")

    def test_budget_enforced_and_never_exceeded(self, client, db, api_key, courses, ai, limits):
        limits(pass_daily_chat=100, pass_budget_usd=0.05)
        token = pass_token(client, api_key)
        reservation = usage.reservation_micro_usd("chat", settings.openai_chat_model)
        expected_ok = 0
        spent = 0
        while spent + reservation <= 50_000:
            spent += CHAT_COST
            expected_ok += 1
        results = [ask(client, token) for _ in range(expected_ok + 3)]
        assert [r.status_code for r in results].count(200) == expected_ok
        assert {r.json()["code"] for r in results if r.status_code != 200} == {"budget_exhausted"}
        p = policy_row(db)
        assert p.spent_micro_usd <= p.budget_micro_usd and p.reserved_micro_usd == 0
        assert ai.calls.count("chat") == expected_ok                # no AI calls once exhausted
        assert status(client, api_key)["budget_exhausted"] is True

    def test_reservation_bounds_cover_the_real_model_output_cap(self):
        from app.agents.llm_factory import get_llm
        assert get_llm().max_tokens == usage.OUTPUT_MAX_TOKENS
        assert usage.reservation_micro_usd("chat", "gpt-4.1-mini") < usage.reservation_micro_usd("chat", "gpt-4o")
        assert usage.model_prices("some-future-model") == usage.model_prices("gpt-4o")   # unknown → priciest


# ── 8. Concurrency ─────────────────────────────────────────────────────────

class TestConcurrency:
    def _concurrent_reserves(self, policy_id, n):
        async def run():
            engine = create_async_engine(settings.postgres_dsn, poolclass=NullPool)
            Session_ = async_sessionmaker(engine, expire_on_commit=False)

            async def one():
                async with Session_() as s:
                    try:
                        await usage.reserve(s, policy_id, "chat", "js")
                        return "ok"
                    except usage.UsagePolicyError as e:
                        return e.code
            try:
                return await asyncio.gather(*[one() for _ in range(n)])
            finally:
                await engine.dispose()
        return asyncio.run(run())

    def test_concurrent_requests_cannot_exceed_daily_quota(self, client, db, api_key, courses, limits):
        limits(pass_daily_chat=5)
        pass_token(client, api_key)
        results = self._concurrent_reserves(policy_row(db).id, 20)
        assert results.count("ok") == 5 and set(results) - {"ok"} == {"daily_limit_reached"}
        p = policy_row(db)
        assert db.scalars(select(UsageCounter).where(UsageCounter.policy_id == p.id)).one().used == 5

    def test_concurrent_requests_cannot_exceed_budget(self, client, db, api_key, courses, limits):
        limits(pass_daily_chat=100, pass_budget_usd=0.1)
        pass_token(client, api_key)
        fits = 100_000 // usage.reservation_micro_usd("chat", settings.openai_chat_model)
        results = self._concurrent_reserves(policy_row(db).id, 20)
        assert results.count("ok") == fits and set(results) - {"ok"} == {"budget_exhausted"}
        p = policy_row(db)
        assert p.reserved_micro_usd <= p.budget_micro_usd


# ── 11. Daily reset at midnight UK time (incl. daylight saving) ────────────

class TestUkDailyReset:
    @pytest.mark.parametrize("utc,expected_day,expected_reset", [
        ("2026-12-31T23:59:00+00:00", "2026-12-31", "2027-01-01T00:00:00+00:00"),   # GMT
        ("2027-06-30T22:59:00+00:00", "2027-06-30", "2027-06-30T23:00:00+00:00"),   # BST: 23:59 local
        ("2027-06-30T23:00:00+00:00", "2027-07-01", "2027-07-01T23:00:00+00:00"),   # BST: 00:00 local
        ("2027-03-28T12:00:00+00:00", "2027-03-28", "2027-03-28T23:00:00+00:00"),   # clocks go forward
        ("2026-10-25T12:00:00+00:00", "2026-10-25", "2026-10-26T00:00:00+00:00"),   # clocks go back
    ])
    def test_london_calendar(self, utc, expected_day, expected_reset):
        now = datetime.fromisoformat(utc)
        assert usage.local_date(now, "Europe/London").isoformat() == expected_day
        assert usage.next_reset(now, "Europe/London").isoformat() == expected_reset

    @pytest.mark.parametrize("before,after", [
        ("2026-12-31T23:58:00+00:00", "2027-01-01T00:01:00+00:00"),   # winter (GMT)
        ("2027-06-30T22:58:00+00:00", "2027-06-30T23:01:00+00:00"),   # summer: UK midnight is 23:00 UTC
    ])
    def test_allowance_resets_at_uk_midnight(self, client, api_key, courses, ai, limits, monkeypatch, before, after):
        limits(pass_daily_chat=2)
        clock = {"now": datetime.fromisoformat(before)}
        monkeypatch.setattr(usage, "_now", lambda: clock["now"])
        token = pass_token(client, api_key)
        assert ask(client, token).status_code == 200 and ask(client, token).status_code == 200
        assert ask(client, token).json()["code"] == "daily_limit_reached"
        clock["now"] = datetime.fromisoformat(after)
        assert ask(client, token).status_code == 200                # a new UK day
