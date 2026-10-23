"""create_all_tables() is safe to run from several workers at once (advisory-lock serialised)."""
import asyncio

from app.db.engine import create_all_tables, engine
from tests.test_v1_documents import pytestmark, sync_engine  # noqa: F401 — real Postgres test DB


def test_concurrent_create_all_is_safe(sync_engine):
    async def run():
        try:
            await asyncio.gather(*[create_all_tables() for _ in range(4)])
        finally:
            await engine.dispose()
    asyncio.run(run())   # raises if the lock SQL is invalid or the workers collide
