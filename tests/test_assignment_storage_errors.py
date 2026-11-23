"""Assignment uploads: storage problems must surface as clear HTTP errors, not a bare 500."""
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api import assignments
from app.storage.cloudinary_client import StorageNotConfigured


@pytest.mark.asyncio
async def test_store_passes_through_the_upload_result(monkeypatch):
    ok = {"public_id": "p", "secure_url": "https://x/p", "bytes": 3}
    monkeypatch.setattr(assignments.storage, "upload_document", AsyncMock(return_value=ok))
    assert await assignments._store(b"abc", "a.pdf", "course", "doc") == ok


@pytest.mark.asyncio
async def test_store_maps_missing_storage_config_to_503(monkeypatch):
    monkeypatch.setattr(assignments.storage, "upload_document", AsyncMock(side_effect=StorageNotConfigured("no cloudinary")))
    with pytest.raises(HTTPException) as e:
        await assignments._store(b"abc", "a.pdf", "course", "doc")
    assert e.value.status_code == 503


@pytest.mark.asyncio
async def test_store_maps_other_upload_failures_to_502(monkeypatch):
    monkeypatch.setattr(assignments.storage, "upload_document", AsyncMock(side_effect=RuntimeError("cloudinary down")))
    with pytest.raises(HTTPException) as e:
        await assignments._store(b"abc", "a.pdf", "course", "doc")
    assert e.value.status_code == 502
