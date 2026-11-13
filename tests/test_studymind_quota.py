"""Quota guardrails; integration/concurrency tests require PostgreSQL."""
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from app.usage.studymind_quota import _lock_id, reserve_paid_feature


def test_advisory_lock_key_is_stable_and_user_specific():
    a, b = uuid4(), uuid4()
    assert _lock_id(a) == _lock_id(a)
    assert _lock_id(a) != _lock_id(b)
    assert -(2 ** 63) <= _lock_id(a) < 2 ** 63


@pytest.mark.asyncio
async def test_unknown_feature_is_rejected_before_database_access():
    db = AsyncMock()
    with pytest.raises(ValueError, match='Unsupported AI feature'):
        await reserve_paid_feature(db, uuid4(), 'unlimited')
    db.execute.assert_not_awaited()


def test_quota_rollout_is_disabled_by_default():
    from app.config import Settings
    assert Settings.model_fields["studymind_quota_enabled"].default is False


def test_generation_routes_use_opt_in_guard():
    from app.api import chat, features
    from inspect import getsource
    assert 'metered_generation(current_user.id, "chat")' in getsource(chat)
    for feature in ("quiz", "flashcards", "summary"):
        assert f'metered_generation(current_user.id, "{feature}")' in getsource(features)


def test_launch_allowances_and_price():
    from app.usage.studymind_quota import (
        FREE_LIMITS, PLUS_LIMITS, PLUS_PRICE_GBP_PENCE, PLUS_DURATION_DAYS,
    )
    assert FREE_LIMITS == {"chat": 10, "quiz": 2, "flashcards": 2, "summary": 1}
    assert PLUS_LIMITS == {"chat": 200, "quiz": 30, "flashcards": 30, "summary": 15}
    assert PLUS_PRICE_GBP_PENCE == 399
    assert PLUS_DURATION_DAYS == 28
