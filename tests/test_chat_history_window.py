"""Core chat context window: _load_history must return the MOST RECENT 20 messages, oldest first."""
import asyncio
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.chat import _load_history
from app.config import get_settings
from app.db.models import ChatMessage, ChatSession
from tests.test_v1_documents import db, pytestmark, sync_engine  # noqa: F401 — fixtures / DB skip


def test_load_history_returns_latest_20_in_order(db):
    session = ChatSession(id=uuid.uuid4(), title="t")
    db.add(session)
    start = datetime.now(timezone.utc)
    for i in range(30):
        db.add(ChatMessage(session_id=session.id, role="user" if i % 2 == 0 else "assistant",
                           content=f"m{i}", created_at=start + timedelta(seconds=i)))
    db.commit()

    async def load():
        engine = create_async_engine(get_settings().postgres_dsn, poolclass=NullPool)
        async with async_sessionmaker(engine)() as s:
            return await _load_history(session.id, s)

    history = asyncio.run(load())
    assert [m["content"] for m in history] == [f"m{i}" for i in range(10, 30)]
