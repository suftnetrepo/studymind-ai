"""Cloudinary storage: private uploads, signed expiring download links, legacy public files."""
import asyncio
from urllib.parse import parse_qs, urlparse

import pytest

from app.config import get_settings
from app.storage import cloudinary_client as cc

PRIVATE_URL = "https://res.cloudinary.com/demo/raw/private/v1/studymind/documents/l/c1/id_notes.pdf"
PUBLIC_URL  = "https://res.cloudinary.com/demo/raw/upload/v1/studymind/documents/l/c1/id_notes.pdf"


@pytest.fixture
def configured(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "cloudinary_cloud_name", "demo")
    monkeypatch.setattr(settings, "cloudinary_api_key", "123")
    monkeypatch.setattr(settings, "cloudinary_api_secret", "secret")


def test_delivery_type_from_url():
    assert cc.delivery_type(PRIVATE_URL) == "private"
    assert cc.delivery_type(PUBLIC_URL) == "upload"
    assert cc.delivery_type("") == "upload"


def test_private_file_gets_signed_expiring_link(configured):
    url = cc.download_url("studymind/documents/l/c1/id_notes.pdf", PRIVATE_URL, ttl=600)
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    assert parsed.netloc == "api.cloudinary.com" and parsed.path == "/v1_1/demo/raw/download"
    assert q["type"] == ["private"] and q["public_id"] == ["studymind/documents/l/c1/id_notes.pdf"]
    assert "signature" in q and int(q["expires_at"][0]) > int(q["timestamp"][0])


def test_legacy_public_file_keeps_stored_url():
    assert cc.download_url("x", PUBLIC_URL) == PUBLIC_URL  # no signing (works without config)


def test_upload_is_private(configured, monkeypatch):
    seen = {}
    monkeypatch.setattr(cc.cloudinary.uploader, "upload",
                        lambda f, **kw: seen.update(kw) or {"public_id": kw["public_id"], "secure_url": PRIVATE_URL, "bytes": 3})
    asyncio.run(cc.upload_document(b"abc", "notes.pdf", "learnify", "c1", "id"))
    assert seen["type"] == "private" and seen["resource_type"] == "raw"


@pytest.mark.parametrize("url,expected", [(PRIVATE_URL, "private"), (PUBLIC_URL, "upload")])
def test_delete_uses_the_files_type(configured, monkeypatch, url, expected):
    seen = {}
    monkeypatch.setattr(cc.cloudinary.uploader, "destroy", lambda pid, **kw: seen.update(kw) or {"result": "ok"})
    assert asyncio.run(cc.delete_document("pid", url)) is True
    assert seen["type"] == expected
