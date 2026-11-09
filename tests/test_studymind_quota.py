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


def test_quota_is_not_implicitly_enabled_on_api_routes():
    from app.api import chat, features
    from inspect import getsource
    assert 'reserve_paid_feature' not in getsource(chat)
    assert 'reserve_paid_feature' not in getsource(features)
