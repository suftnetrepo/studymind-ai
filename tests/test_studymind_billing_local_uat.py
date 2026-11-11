"""Billing and allowance behaviour for local UAT (Stripe TEST mode only, no network, no real Stripe).

Part 1: pure configuration guards (always run).
Part 2: database-backed checks of the real entitlement and quota code: replayed/concurrent webhooks, expiry, free and Plus limits,
        concurrency, no fallback from an exhausted Plus pass (skipped if the local test database is unreachable).
"""
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, delete, select
from sqlalchemy.pool import NullPool

from app.billing import stripe_checkout as sc
from app.config import get_settings

settings = get_settings()


def _s(**kw):
    base = dict(studymind_billing_enabled=True, studymind_stripe_secret_key="sk_test_x" * 3, studymind_stripe_webhook_secret="whsec_x" * 3,
                studymind_checkout_success_url="https://app.example.com/ok", studymind_checkout_cancel_url="https://app.example.com/no",
                studymind_stripe_api_base="https://api.stripe.com", app_env="development")
    base.update(kw)
    return SimpleNamespace(**base)


# ── Part 1: configuration guards ──────────────────────────────────────────

def test_real_stripe_api_is_the_default():
    assert settings.studymind_stripe_api_base == "https://api.stripe.com"
    with patch.object(sc, "get_settings", return_value=_s()):
        assert sc.stripe_api_base() == "https://api.stripe.com"


@pytest.mark.parametrize("base,env,ok", [
    ("http://127.0.0.1:18400", "development", True),
    ("http://localhost:18400", "test", True),
    ("http://127.0.0.1:18400", "staging", False),        # never outside local development/test
    ("http://127.0.0.1:18400", "production", False),
    ("https://evil.example.com", "development", False),  # a non-loopback host can never stand in for Stripe
    ("https://api.stripe.com.evil.example.com", "development", False),
])
def test_stripe_api_override_is_loopback_and_development_only(base, env, ok):
    with patch.object(sc, "get_settings", return_value=_s(studymind_stripe_api_base=base, app_env=env)):
        if ok:
            assert sc.stripe_api_base() == base
        else:
            with pytest.raises(RuntimeError):
                sc.stripe_api_base()
        assert (sc.checkout_configured() is True) == ok


@pytest.mark.parametrize("url,env,ok", [
    ("http://localhost:3100/my-learning/plans", "development", True),
    ("http://127.0.0.1:3100/x", "test", True),
    ("http://localhost:3100/x", "staging", False),
    ("http://localhost:3100/x", "production", False),
    ("http://app.example.com/x", "development", False),   # plain http to a real host is never accepted
    ("https://app.example.com/x", "production", True),
    ("javascript:alert(1)", "development", False),
])
def test_return_urls_need_https_except_loopback_in_development(url, env, ok):
    with patch.object(sc, "get_settings", return_value=_s(studymind_checkout_success_url=url, app_env=env)):
        assert sc.checkout_configured() is ok


@pytest.mark.parametrize("key", ["sk_live_abc", "rk_live_abc", "pk_test_abc", "", "sk_prod"])
def test_non_test_secret_keys_fail_closed(key):
    with patch.object(sc, "get_settings", return_value=_s(studymind_stripe_secret_key=key)):
        assert sc.checkout_configured() is False


# ── Part 2: database-backed behaviour ─────────────────────────────────────

def _db_available() -> bool:
    try:
        eng = create_engine(settings.postgres_dsn_sync, poolclass=NullPool)
        with eng.connect():
            pass
        eng.dispose()
        return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_available(), reason=f"Postgres test database '{settings.postgres_db}' not reachable")


_maker = None   # per-test sessionmaker: the app's global engine is bound to one event loop, each test here has its own


def AsyncSessionLocal():   # noqa: N802 - mirrors the app's name so the helpers read the same
    return _maker()


@pytest_asyncio.fixture
async def user():
    global _maker
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from app.db.engine import Base
    from app.db.models import User
    assert "test" in settings.postgres_db, "refusing to run against a non-test database"
    engine = create_async_engine(settings.postgres_dsn, poolclass=NullPool)
    _maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    uid = uuid.uuid4()
    async with AsyncSessionLocal() as db:
        db.add(User(id=uid, email=f"billing-{uid.hex[:8]}@test.dev", password_hash="x", full_name="Billing Test", role="self_learner"))
        await db.commit()
    yield uid
    async with AsyncSessionLocal() as db:
        await db.execute(delete(User).where(User.id == uid))
        await db.commit()
    await engine.dispose()


def _paid_session(uid, sid=None, pi=None):
    sid = sid or f"cs_test_{uuid.uuid4().hex}"
    return {"id": sid, "livemode": False, "mode": "payment", "payment_status": "paid", "status": "complete", "currency": "gbp",
            "amount_total": 399, "client_reference_id": str(uid), "metadata": {"studymind_user_id": str(uid), "studymind_product": "plus_28d_v1"},
            "payment_intent": pi or f"pi_{uuid.uuid4().hex}"}


async def _activate(session):
    from app.billing.stripe_webhook import activate_paid_checkout
    async with AsyncSessionLocal() as db:
        async with db.begin():
            return await activate_paid_checkout(db, session)


async def _entitlements(uid):
    from app.db.models import StudyMindEntitlement
    async with AsyncSessionLocal() as db:
        return (await db.execute(select(StudyMindEntitlement).where(StudyMindEntitlement.user_id == uid))).scalars().all()


async def _reserve(uid, feature):
    from app.usage.studymind_quota import reserve_feature
    async with AsyncSessionLocal() as db:
        async with db.begin():
            return await reserve_feature(db, uid, feature)


async def _use_until_refused(uid, feature, cap=300):
    from app.usage.studymind_quota import StudyMindQuotaExceeded
    ok = 0
    for _ in range(cap):
        try:
            await _reserve(uid, feature)
            ok += 1
        except StudyMindQuotaExceeded:
            break
    return ok


@needs_db
@pytest.mark.asyncio
async def test_free_limits_are_10_2_2_1(user):
    got = {f: await _use_until_refused(user, f) for f in ("chat", "quiz", "flashcards", "summary")}
    assert got == {"chat": 10, "quiz": 2, "flashcards": 2, "summary": 1}


@needs_db
@pytest.mark.asyncio
async def test_plus_limits_are_200_30_30_15_and_start_after_payment(user):
    assert await _activate(_paid_session(user)) is True
    ent = (await _entitlements(user))[0]
    assert ent.status == "active" and ent.expires_at - ent.starts_at == timedelta(days=28)
    got = {f: await _use_until_refused(user, f) for f in ("chat", "quiz", "flashcards", "summary")}
    assert got == {"chat": 200, "quiz": 30, "flashcards": 30, "summary": 15}


@needs_db
@pytest.mark.asyncio
async def test_exhausted_plus_does_not_silently_fall_back_to_free(user):
    await _activate(_paid_session(user))
    assert await _use_until_refused(user, "summary") == 15
    assert await _use_until_refused(user, "summary") == 0          # still refused, free's allowance is not used as a bonus


@needs_db
@pytest.mark.asyncio
async def test_replayed_webhook_grants_one_entitlement(user):
    session = _paid_session(user)
    assert await _activate(session) is True
    assert await _activate(session) is False
    assert await _activate(dict(session)) is False
    assert len(await _entitlements(user)) == 1


@needs_db
@pytest.mark.asyncio
async def test_concurrent_deliveries_of_one_payment_grant_one_entitlement(user):
    session = _paid_session(user)
    results = await asyncio.gather(*[_activate(dict(session)) for _ in range(8)])
    assert results.count(True) == 1 and results.count(False) == 7
    assert len(await _entitlements(user)) == 1


@needs_db
@pytest.mark.asyncio
async def test_unpaid_or_wrong_amount_or_live_sessions_grant_nothing(user):
    from app.billing.stripe_webhook import InvalidStripeWebhook
    for change in ({"payment_status": "unpaid"}, {"status": "open"}, {"amount_total": 1}, {"currency": "usd"}, {"livemode": True},
                   {"mode": "subscription"}, {"payment_intent": None}):
        bad = {**_paid_session(user), **change}
        with pytest.raises(InvalidStripeWebhook):
            await _activate(bad)
    assert await _entitlements(user) == []


@needs_db
@pytest.mark.asyncio
async def test_a_checkout_for_user_a_cannot_activate_user_b(user):
    other = uuid.uuid4()
    forged = {**_paid_session(user), "client_reference_id": str(other)}
    from app.billing.stripe_webhook import InvalidStripeWebhook
    with pytest.raises(InvalidStripeWebhook):
        await _activate(forged)


@needs_db
@pytest.mark.asyncio
async def test_expired_pass_returns_the_user_to_free(user):
    from app.db.models import StudyMindEntitlement
    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        db.add(StudyMindEntitlement(user_id=user, stripe_checkout_session_id=f"cs_test_{uuid.uuid4().hex}", status="active",
                                    feature_limits={"chat": 200, "quiz": 30, "flashcards": 30, "summary": 15},
                                    starts_at=now - timedelta(days=29), expires_at=now - timedelta(minutes=1)))
        await db.commit()
    assert (await _reserve(user, "chat"))["plan"] == "free"
    async with AsyncSessionLocal() as db:
        assert await sc.has_active_plus(db, user) is False


@needs_db
@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["suspended", "revoked"])
async def test_reversed_pass_is_not_active(user, status):
    from app.db.models import StudyMindEntitlement
    await _activate(_paid_session(user))
    async with AsyncSessionLocal() as db:
        ent = (await db.execute(select(StudyMindEntitlement).where(StudyMindEntitlement.user_id == user))).scalar_one()
        ent.status = status
        await db.commit()
        assert await sc.has_active_plus(db, user) is False
    assert (await _reserve(user, "chat"))["plan"] == "free"


@needs_db
@pytest.mark.asyncio
async def test_active_pass_blocks_a_second_checkout(user):
    from fastapi import HTTPException
    from app.api.billing import checkout
    with patch("app.api.billing.checkout_configured", return_value=True), patch("app.api.billing.AsyncSessionLocal", AsyncSessionLocal):
        await _activate(_paid_session(user))
        with pytest.raises(HTTPException) as exc:
            await checkout(SimpleNamespace(id=user, email="x@test.dev"))
    assert exc.value.status_code == 409


@needs_db
@pytest.mark.asyncio
async def test_simultaneous_requests_cannot_overspend_the_free_allowance(user):
    from app.usage.studymind_quota import StudyMindQuotaExceeded

    async def one():
        try:
            await _reserve(user, "chat")
            return True
        except StudyMindQuotaExceeded:
            return False
    results = await asyncio.gather(*[one() for _ in range(25)])
    assert results.count(True) == 10 and results.count(False) == 15
