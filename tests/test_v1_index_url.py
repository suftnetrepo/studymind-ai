"""
POST /api/v1/documents/index-url — add a course document from a URL without re-uploading it.

Endpoint tests use the real Postgres test DB and fakes from test_v1_documents (the download is
faked too). The downloader's SSRF guards are unit-tested with a fake resolver + mock transport.
"""
import asyncio
import uuid

import httpx
import pytest
from sqlalchemy import func, select

from app.api.v1 import documents as v1_documents
from app.db.models import Document, PlatformDocument
from app.storage import url_fetch
from app.storage.url_fetch import FetchedFile, UrlFetchError, fetch_document
from tests.test_v1_documents import (  # noqa: F401 — pytest fixtures
    api_key, auth, client, db, fakes, session_token, sync_engine,
)

CLOUDINARY_URL = "https://res.cloudinary.com/demo/raw/upload/v1712/learnify/lectures/week1_notes.pdf"
OTHER_URL      = "https://files.example.org/course/Intro%20Guide.docx"


@pytest.fixture
def downloads(monkeypatch):
    """Fake downloader: returns `downloads.content` (+ content type), records the URLs."""
    class Downloads:
        calls:        list = []
        content:      bytes = b"Line one\nLine two\nLine three\n"
        content_type: str = "application/pdf"
        error:        str | None = None

    d = Downloads()
    d.calls = []

    async def fake_fetch(url, max_bytes, **kw):
        d.calls.append(url)
        if d.error:
            raise UrlFetchError(d.error)
        return FetchedFile(content=d.content, content_type=d.content_type, final_url=url)

    monkeypatch.setattr(v1_documents, "fetch_document", fake_fetch)
    return d


def index_url(client, token, url, **body):
    payload = {"course_id": "c1", "user_id": "tutor_1", "url": url, **body}
    return client.post("/api/v1/documents/index-url", headers=auth(token),
                       json={k: v for k, v in payload.items() if v is not None})


class TestIndexUrl:
    def test_cloudinary_url_indexes_without_reupload(self, client, db, api_key, fakes, downloads):
        r = index_url(client, api_key, CLOUDINARY_URL)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "indexing" and body["filename"] == "week1_notes.pdf"
        assert body["url"] == CLOUDINARY_URL            # links to the original, no signed copy
        assert downloads.calls == [CLOUDINARY_URL]
        assert fakes.uploads == []                        # nothing re-uploaded to our storage

        # Background indexing ran into the course module
        pdoc = db.get(PlatformDocument, uuid.UUID(body["id"]))
        assert pdoc.status == "ready" and pdoc.chunk_count == 3
        assert pdoc.cloudinary_url == CLOUDINARY_URL
        assert pdoc.cloudinary_public_id.startswith(v1_documents.EXTERNAL_PREFIX)
        assert pdoc.uploaded_by == "tutor_1" and pdoc.file_format == "pdf"
        doc = db.get(Document, pdoc.document_id)
        assert doc.status == "indexed" and doc.doc_metadata["source"] == "platform_url"

    def test_other_url_indexes(self, client, db, api_key, fakes, downloads):
        downloads.content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        r = index_url(client, api_key, OTHER_URL)
        assert r.status_code == 202, r.text
        assert r.json()["filename"] == "Intro Guide.docx"   # URL-decoded
        pdoc = db.get(PlatformDocument, uuid.UUID(r.json()["id"]))
        assert pdoc.status == "ready" and pdoc.file_format == "docx"

    def test_listed_with_uploads(self, client, api_key, fakes, downloads):
        index_url(client, api_key, OTHER_URL)
        docs = client.get("/api/v1/documents/list", headers=auth(api_key), params={"course_id": "c1"}).json()
        assert [(d["filename"], d["status"], d["url"]) for d in docs] == [("Intro Guide.docx", "ready", OTHER_URL)]

    def test_filename_given_is_used(self, client, api_key, fakes, downloads):
        r = index_url(client, api_key, CLOUDINARY_URL, filename="Week 1 notes.pdf")
        assert r.json()["filename"] == "Week 1 notes.pdf"

    def test_type_from_content_type_when_url_has_no_extension(self, client, api_key, fakes, downloads):
        downloads.content_type = "text/plain"
        r = index_url(client, api_key, "https://example.org/download?id=42")
        assert r.status_code == 202, r.text
        assert r.json()["filename"] == "download.txt"

    def test_already_indexed_returns_existing(self, client, db, api_key, fakes, downloads):
        first = index_url(client, api_key, CLOUDINARY_URL).json()
        r = index_url(client, api_key, CLOUDINARY_URL)
        assert r.status_code == 202
        assert r.json()["already_indexed"] is True and r.json()["id"] == first["id"]
        assert r.json()["message"] == "This URL has already been added"
        assert len(downloads.calls) == 1
        assert db.scalar(select(func.count()).select_from(PlatformDocument)) == 1

    def test_same_url_other_course_is_separate(self, client, db, api_key, fakes, downloads):
        index_url(client, api_key, CLOUDINARY_URL)
        index_url(client, api_key, CLOUDINARY_URL, course_id="c2")
        assert db.scalar(select(func.count()).select_from(PlatformDocument)) == 2

    def test_failed_url_can_be_retried(self, client, db, api_key, fakes, downloads):
        downloads.content = b"FAIL"                       # FakeIngestor fails on this
        first = index_url(client, api_key, CLOUDINARY_URL).json()
        assert db.get(PlatformDocument, uuid.UUID(first["id"])).status == "failed"
        downloads.content = b"Readable now\n"
        again = index_url(client, api_key, CLOUDINARY_URL).json()
        assert again["id"] != first["id"] and "already_indexed" not in again

    @pytest.mark.parametrize("url", ["http://example.org/notes.pdf", "ftp://example.org/notes.pdf", "notes.pdf"])
    def test_non_https_400(self, client, api_key, fakes, downloads, url):
        r = index_url(client, api_key, url)
        assert r.status_code == 400 and r.json()["detail"] == "URL must start with https://"
        assert downloads.calls == []

    @pytest.mark.parametrize("url", ["https://example.org/photo.jpg", "https://example.org/page.html",
                                     "https://example.org/setup.exe?x=1"])
    def test_unsupported_type_400_before_download(self, client, api_key, fakes, downloads, url):
        r = index_url(client, api_key, url)
        assert r.status_code == 400 and "not supported" in r.json()["detail"]
        assert downloads.calls == []

    def test_unknown_type_without_extension_400(self, client, api_key, fakes, downloads):
        downloads.content_type = "text/html"
        r = index_url(client, api_key, "https://example.org/download?id=42")
        assert r.status_code == 400 and "Couldn't tell the file type" in r.json()["detail"]

    def test_download_error_400_with_reason(self, client, db, api_key, fakes, downloads):
        downloads.error = "The file isn't publicly accessible (access denied)"
        r = index_url(client, api_key, CLOUDINARY_URL)
        assert r.status_code == 400 and r.json()["detail"] == downloads.error
        assert db.scalar(select(func.count()).select_from(PlatformDocument)) == 0

    def test_unauthorized_401(self, client, downloads):
        r = client.post("/api/v1/documents/index-url", json={"course_id": "c1", "user_id": "u", "url": CLOUDINARY_URL})
        assert r.status_code == 401
        assert downloads.calls == []

    def test_tutor_session_token(self, client, db, api_key, fakes, downloads):
        token = session_token(client, api_key, course_id="c9", user_id="tutor_9", role="tutor")
        r = client.post("/api/v1/documents/index-url", headers=auth(token), json={"url": CLOUDINARY_URL})
        assert r.status_code == 202, r.text
        pdoc = db.get(PlatformDocument, uuid.UUID(r.json()["id"]))
        assert (pdoc.platform_course_id, pdoc.uploaded_by) == ("c9", "tutor_9")   # from the token

    def test_student_session_token_403(self, client, api_key, fakes, downloads):
        token = session_token(client, api_key, role="student", user_id="stu_1")
        r = client.post("/api/v1/documents/index-url", headers=auth(token), json={"url": CLOUDINARY_URL})
        assert r.status_code == 403
        assert downloads.calls == []


class TestLinkedDocumentLifecycle:
    def test_delete_never_deletes_the_original_file(self, client, db, api_key, fakes, downloads):
        doc_id = index_url(client, api_key, CLOUDINARY_URL).json()["id"]
        r = client.delete(f"/api/v1/documents/{doc_id}", headers=auth(api_key), params={"course_id": "c1"})
        assert r.status_code == 200
        assert fakes.deletes == []                        # the tutor's Cloudinary file is untouched
        assert db.scalar(select(func.count()).select_from(PlatformDocument)) == 0

    def test_replace_uploads_new_file_but_keeps_original(self, client, db, api_key, fakes, downloads):
        doc_id = index_url(client, api_key, CLOUDINARY_URL).json()["id"]
        r = client.post(f"/api/v1/documents/{doc_id}/replace", headers=auth(api_key), data={"course_id": "c1"},
                        files={"file": ("v2.pdf", b"New\n", "application/pdf")})
        assert r.status_code == 202, r.text
        assert fakes.deletes == []                        # original not deleted
        pdoc = db.get(PlatformDocument, uuid.UUID(doc_id))
        assert not v1_documents.is_external(pdoc) and pdoc.filename == "v2.pdf"


# ── Downloader (SSRF guards) ───────────────────────────────────────────────

PUBLIC = lambda host: ["93.184.216.34"]                      # noqa: E731


def fetch(url, **kw):
    return asyncio.run(fetch_document(url, 1024, **kw))


def transport(handler):
    return httpx.MockTransport(handler)


def ok_pdf(request):
    return httpx.Response(200, content=b"%PDF-1.4 data", headers={"content-type": "application/pdf; charset=binary"})


class TestFetchDocument:
    def test_downloads_public_file(self):
        f = fetch("https://example.org/a.pdf", resolve=PUBLIC, transport=transport(ok_pdf))
        assert f.content == b"%PDF-1.4 data" and f.content_type == "application/pdf"

    @pytest.mark.parametrize("addr", ["127.0.0.1", "10.0.0.5", "192.168.1.10", "172.16.0.1",
                                      "169.254.169.254", "::1", "fd00::1", "0.0.0.0"])
    def test_private_addresses_blocked(self, addr):
        with pytest.raises(UrlFetchError, match="isn't publicly reachable"):
            fetch("https://internal.example/a.pdf", resolve=lambda h: [addr],
                                 transport=transport(ok_pdf))

    def test_redirect_to_private_address_blocked(self):
        def handler(request):
            if request.url.host == "example.org":
                return httpx.Response(302, headers={"location": "https://metadata.internal/latest"})
            return ok_pdf(request)
        resolve = lambda h: ["169.254.169.254"] if h == "metadata.internal" else ["93.184.216.34"]  # noqa: E731
        with pytest.raises(UrlFetchError, match="isn't publicly reachable"):
            fetch("https://example.org/a.pdf", resolve=resolve, transport=transport(handler))

    def test_redirect_to_http_blocked(self):
        handler = lambda r: httpx.Response(301, headers={"location": "http://example.org/a.pdf"})  # noqa: E731
        with pytest.raises(UrlFetchError, match="https"):
            fetch("https://example.org/a.pdf", resolve=PUBLIC, transport=transport(handler))

    def test_follows_public_redirect(self):
        def handler(request):
            if request.url.path == "/old.pdf":
                return httpx.Response(302, headers={"location": "/new.pdf"})
            return ok_pdf(request)
        f = fetch("https://example.org/old.pdf", resolve=PUBLIC, transport=transport(handler))
        assert f.final_url == "https://example.org/new.pdf"

    def test_too_many_redirects(self):
        handler = lambda r: httpx.Response(302, headers={"location": "/loop"})  # noqa: E731
        with pytest.raises(UrlFetchError, match="Too many redirects"):
            fetch("https://example.org/a.pdf", resolve=PUBLIC, transport=transport(handler))

    def test_size_cap(self):
        big = lambda r: httpx.Response(200, content=b"x" * 2048)  # noqa: E731
        with pytest.raises(UrlFetchError, match="too large"):
            fetch("https://example.org/a.pdf", resolve=PUBLIC, transport=transport(big))

    @pytest.mark.parametrize("status,msg", [(403, "isn't publicly accessible"), (401, "isn't publicly accessible"),
                                            (404, "No file was found"), (500, "HTTP 500")])
    def test_http_errors(self, status, msg):
        with pytest.raises(UrlFetchError, match=msg):
            fetch("https://example.org/a.pdf", resolve=PUBLIC,
                                 transport=transport(lambda r: httpx.Response(status)))

    def test_credentials_in_url_rejected(self):
        with pytest.raises(UrlFetchError, match="username or password"):
            fetch("https://user:pw@example.org/a.pdf", resolve=PUBLIC, transport=transport(ok_pdf))

    def test_unknown_host(self):
        def fail(host):
            raise url_fetch.socket.gaierror("nope")
        with pytest.raises(UrlFetchError, match="Couldn't find the server"):
            fetch("https://no-such-host.invalid/a.pdf", resolve=fail, transport=transport(ok_pdf))

    def test_filename_from_url(self):
        assert url_fetch.filename_from_url(OTHER_URL) == "Intro Guide.docx"
        assert url_fetch.filename_from_url("https://example.org/") == ""
