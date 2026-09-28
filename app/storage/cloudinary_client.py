"""
Cloudinary storage for platform-uploaded course documents.

Files are stored as `raw` resources under
    {CLOUDINARY_FOLDER}/{platform}/{course_id}/{document_id}_{filename}
The document UUID in the public_id keeps same-named uploads from overwriting each other.

New uploads are `private`: their stored URL can't be fetched, so course materials aren't
public. Viewers get a signed, expiring download link from `download_url()` instead. Files
uploaded before this change are `upload` type (public) and keep working as-is — the type is
read from the stored URL (`/raw/private/…` vs `/raw/upload/…`).
"""
from __future__ import annotations

import io
import re
import time

import cloudinary
import cloudinary.uploader
import cloudinary.utils
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.logging_config import get_logger

log = get_logger(__name__)


class StorageNotConfigured(RuntimeError):
    pass


def _configure() -> None:
    settings = get_settings()
    if not settings.cloudinary_configured:
        raise StorageNotConfigured("Cloudinary is not configured (CLOUDINARY_* env vars)")
    cloudinary.config(
        cloud_name=settings.cloudinary_cloud_name,
        api_key=settings.cloudinary_api_key,
        api_secret=settings.cloudinary_api_secret,
        secure=True,
    )


PRIVATE = "private"
DOWNLOAD_LINK_TTL_SECONDS = 3600


def delivery_type(stored_url: str) -> str:
    """'private' for private uploads, 'upload' for older public ones."""
    return PRIVATE if "/raw/private/" in (stored_url or "") else "upload"


def download_url(public_id: str, stored_url: str, ttl: int = DOWNLOAD_LINK_TTL_SECONDS) -> str:
    """
    A URL a browser can open. Private files get a signed link that expires after `ttl`
    seconds; older public files return their stored URL.
    """
    if delivery_type(stored_url) != PRIVATE:
        return stored_url
    _configure()
    return cloudinary.utils.private_download_url(
        public_id, "",                    # raw public_ids already include the extension
        resource_type="raw", type=PRIVATE, expires_at=int(time.time()) + ttl,
    )


def _safe_segment(value: str, max_len: int = 100) -> str:
    """Restrict to characters that are safe in a Cloudinary public_id path segment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:max_len] or "file"


def build_public_id(platform: str, course_id: str, document_id: str, filename: str) -> str:
    folder = get_settings().cloudinary_folder.strip("/")
    # Raw resources keep the extension in the public_id
    return f"{folder}/{_safe_segment(platform)}/{_safe_segment(course_id)}/{document_id}_{_safe_segment(filename, 150)}"


async def upload_document(
    file_bytes:  bytes,
    filename:    str,
    platform:    str,
    course_id:   str,
    document_id: str,
) -> dict:
    """Upload bytes as a raw resource. Returns {public_id, secure_url, bytes, format}."""
    _configure()
    public_id = build_public_id(platform, course_id, document_id, filename)

    result = await run_in_threadpool(
        cloudinary.uploader.upload,
        io.BytesIO(file_bytes),
        public_id=public_id,
        resource_type="raw",
        type=PRIVATE,
        overwrite=True,
        filename_override=filename,
    )
    log.info("cloudinary_upload_success", public_id=result["public_id"], bytes=result.get("bytes", 0))
    return {
        "public_id":  result["public_id"],
        "secure_url": result["secure_url"],
        "bytes":      result.get("bytes", len(file_bytes)),
        # Raw resources have no "format" — fall back to the extension
        "format":     result.get("format") or filename.rsplit(".", 1)[-1].lower(),
    }


async def delete_document(public_id: str, stored_url: str = "") -> bool:
    """Delete a raw resource. Never raises — storage cleanup must not block DB cleanup."""
    try:
        _configure()
        result = await run_in_threadpool(
            cloudinary.uploader.destroy, public_id,
            resource_type="raw", type=delivery_type(stored_url), invalidate=True,
        )
        ok = result.get("result") in ("ok", "not found")
        if not ok:
            log.warning("cloudinary_delete_unexpected", public_id=public_id, result=result)
        return ok
    except Exception as e:
        log.error("cloudinary_delete_failed", public_id=public_id, error=str(e))
        return False
