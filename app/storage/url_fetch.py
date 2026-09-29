"""
Download a course document from a URL a tutor pasted (POST /v1/documents/index-url).

The server fetches whatever URL it's given, so this guards against server-side request forgery:
HTTPS only, no credentials in the URL, and the host must resolve to public addresses — checked
for the first URL and again for every redirect (followed manually, at most MAX_REDIRECTS). The
body is streamed and capped at `max_bytes`.

Residual risk: DNS could change between our check and httpx's own lookup (rebinding). The
checks still block the common cases — localhost, private networks, cloud metadata endpoints.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from starlette.concurrency import run_in_threadpool

MAX_REDIRECTS   = 3
TIMEOUT_SECONDS = 30.0
USER_AGENT      = "StudyMind-Indexer/1.0"


class UrlFetchError(ValueError):
    """The URL can't be used; the message is safe to show the user."""


@dataclass
class FetchedFile:
    content:      bytes
    content_type: str     # e.g. "application/pdf" (parameters stripped, lower-case)
    final_url:    str     # after redirects


Resolver = Callable[[str], list[str]]


def _resolve(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


def _check_url(url: str, resolve: Resolver) -> None:
    parts = urlsplit(url)
    if parts.scheme.lower() != "https":
        raise UrlFetchError("URL must start with https://")
    if not parts.hostname:
        raise UrlFetchError("That doesn't look like a valid URL")
    if parts.username or parts.password:
        raise UrlFetchError("URLs with a username or password aren't supported")
    try:
        addresses = resolve(parts.hostname)
    except (socket.gaierror, UnicodeError):
        raise UrlFetchError(f"Couldn't find the server {parts.hostname}")
    if not addresses:
        raise UrlFetchError(f"Couldn't find the server {parts.hostname}")
    for addr in addresses:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
        if not ip.is_global or ip.is_multicast:
            raise UrlFetchError("That address isn't publicly reachable")


async def fetch_document(
    url:       str,
    max_bytes: int,
    *,
    resolve:   Optional[Resolver] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,   # tests
) -> FetchedFile:
    """Download `url` (following up to MAX_REDIRECTS public redirects). Raises UrlFetchError."""
    resolve_fn = resolve or _resolve
    too_big = f"File too large. Maximum size is {max_bytes // (1024 * 1024)}MB."

    async with httpx.AsyncClient(
        follow_redirects=False, timeout=TIMEOUT_SECONDS, transport=transport,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            await run_in_threadpool(_check_url, current, resolve_fn)   # DNS lookup blocks
            try:
                async with client.stream("GET", current) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise UrlFetchError("The server sent a redirect without a location")
                        current = urljoin(current, location)
                        continue
                    if resp.status_code in (401, 403):
                        raise UrlFetchError("The file isn't publicly accessible (access denied)")
                    if resp.status_code == 404:
                        raise UrlFetchError("No file was found at that URL (404)")
                    if resp.status_code != 200:
                        raise UrlFetchError(f"Couldn't download the file (HTTP {resp.status_code})")

                    declared = resp.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > max_bytes:
                        raise UrlFetchError(too_big)
                    chunks, size = [], 0
                    async for chunk in resp.aiter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise UrlFetchError(too_big)
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    if not content:
                        raise UrlFetchError("The file at that URL is empty")
                    ctype = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    return FetchedFile(content=content, content_type=ctype, final_url=current)
            except httpx.TimeoutException:
                raise UrlFetchError("The server took too long to respond")
            except httpx.HTTPError:
                raise UrlFetchError("Couldn't connect to download the file")
        raise UrlFetchError("Too many redirects")


def filename_from_url(url: str) -> str:
    """Last path segment, URL-decoded ('' when the path has none)."""
    path = urlsplit(url).path
    return unquote(path.rstrip("/").rsplit("/", 1)[-1]) if path else ""


EXTENSION_FOR_TYPE = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
}
