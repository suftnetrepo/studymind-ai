"""
v1 Platform documents (Sprint 3) + shared course modules.

Runs against a real Postgres test database (POSTGRES_DB, `studymind_test` by default — see
conftest.py) so joins, cascades and background indexing use real SQL. Cloudinary, Typesense and
the embedding ingestor are replaced with fakes. Skipped when the database is unreachable.
"""
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from app.api.main import app
from app.api.v1 import documents as v1_documents
from app.api.v1 import platform as v1_platform
from app.auth.api_key_auth import generate_api_key
from app.config import get_settings
from app.db.engine import Base, get_db
from app.db.models import (
    ApiKey, Document, DocumentChunk, Module, ModuleDocument, PlatformCourse, PlatformDocument,
)
from app.storage import cloudinary_client

settings = get_settings()


def _db_available() -> bool:
    try:
        eng = create_engine(settings.postgres_dsn_sync, poolclass=NullPool)
        with eng.connect():
            pass
        eng.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason=f"Postgres test database '{settings.postgres_db}' not reachable",
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sync_engine():
    assert "test" in settings.postgres_db, "refusing to run destructive tests against a non-test DB"
    eng = create_engine(settings.postgres_dsn_sync, poolclass=NullPool)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def db(sync_engine):
    """Sync session for arranging/asserting; tables are truncated after each test."""
    with Session(sync_engine) as session:
        yield session
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
    with sync_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


class FakeIngestor:
    """Splits content into one chunk per non-empty line; fails on content b'FAIL'."""
    def __init__(self):
        self.calls = []

    def ingest(self, filename, content, document_id, **kw):
        self.calls.append({"filename": filename, "document_id": document_id, **kw})
        if content == b"FAIL":
            return {"status": "failed", "chunk_count": 0, "error": "unreadable", "chunks": [], "indexed_at": None}
        lines = [l for l in content.decode("utf-8", "replace").splitlines() if l.strip()]
        chunks = [{"typesense_id": f"{document_id}__{i}", "chunk_index": i, "content": l,
                   "token_count": len(l.split()), "chunk_metadata": {}} for i, l in enumerate(lines)]
        return {"status": "indexed", "chunk_count": len(chunks), "error": None,
                "chunks": chunks, "indexed_at": datetime.now(timezone.utc)}


class Recorder:
    def __init__(self):
        self.uploads, self.deletes, self.ts_deletes, self.superseded = [], [], [], []


@pytest.fixture
def fakes(monkeypatch):
    rec      = Recorder()
    ingestor = FakeIngestor()
    rec.ingestor = ingestor

    async def upload(content, filename, platform, course_id, document_id):
        rec.uploads.append((filename, platform, course_id, document_id))
        public_id = cloudinary_client.build_public_id(platform, course_id, document_id, filename)
        return {"public_id": public_id, "secure_url": f"https://res.cloudinary.com/demo/raw/upload/{public_id}",
                "bytes": len(content), "format": filename.rsplit(".", 1)[-1]}

    async def delete(public_id):
        rec.deletes.append(public_id)
        return True

    monkeypatch.setattr(cloudinary_client, "upload_document", upload)
    monkeypatch.setattr(cloudinary_client, "delete_document", delete)
    monkeypatch.setattr(v1_documents, "_require_storage", lambda: None)
    monkeypatch.setattr(v1_platform, "get_ingestor", lambda: ingestor)
    monkeypatch.setattr(v1_documents, "get_typesense_client", lambda: None)
    monkeypatch.setattr(v1_platform, "get_typesense_client", lambda: None)
    monkeypatch.setattr(v1_documents, "delete_document_chunks", lambda ts, doc_id: rec.ts_deletes.append(doc_id))
    monkeypatch.setattr(v1_platform, "delete_document_chunks", lambda ts, doc_id: rec.ts_deletes.append(doc_id))
    monkeypatch.setattr(v1_documents, "mark_chunks_superseded",
                        lambda ts, doc_id, version: rec.superseded.append((doc_id, version)))
    return rec


@pytest.fixture
def client(sync_engine, monkeypatch):
    # NullPool: TestClient runs the app on its own event loop, so connections must not be reused
    engine  = create_async_engine(settings.postgres_dsn, poolclass=NullPool)
    Session_ = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async def _get_db():
        async with Session_() as session:
            yield session
            await session.commit()

    app.dependency_overrides[get_db] = _get_db
    monkeypatch.setattr(v1_documents, "AsyncSessionLocal", Session_)
    monkeypatch.setattr(v1_platform, "AsyncSessionLocal", Session_)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def api_key(db):
    full_key, key_hash, prefix = generate_api_key()
    key = ApiKey(id=uuid.uuid4(), name="Test", key_hash=key_hash, key_prefix=prefix,
                 platform="learnify", owner_email="t@test.dev", is_active=True, request_count=0)
    db.add(key)
    db.commit()
    return full_key


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def session_token(client, api_key, course_id="c1", user_id="tutor_1", role="tutor") -> str:
    r = client.post("/api/v1/auth/session", headers=auth(api_key),
                    json={"course_id": course_id, "user_id": user_id, "user_role": role})
    assert r.status_code == 201, r.text
    return r.json()["session_token"]


def upload(client, token, filename="notes.pdf", content=b"Line one\nLine two\n",
           course_id="c1", user_id="tutor_1", **kw):
    data = {k: v for k, v in {"course_id": course_id, "user_id": user_id}.items() if v is not None}
    return client.post("/api/v1/documents/upload", headers=auth(token), data=data,
                       files={"file": (filename, content, kw.get("content_type", "application/pdf"))})


# ── Upload ─────────────────────────────────────────────────────────────────

class TestUpload:
    def test_upload_pdf_indexes_into_course_module(self, client, db, api_key, fakes):
        r = upload(client, api_key)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "indexing" and body["filename"] == "notes.pdf"
        assert body["url"].startswith("https://res.cloudinary.com/")

        # Stored under {folder}/{platform}/{course}/{doc id}_{filename}
        [(filename, platform, course, doc_id)] = fakes.uploads
        assert (filename, platform, course, doc_id) == ("notes.pdf", "learnify", "c1", body["id"])

        # Background task has run: indexed into the course module with 2 chunks
        pdoc = db.get(PlatformDocument, uuid.UUID(body["id"]))
        assert pdoc.status == "ready" and pdoc.chunk_count == 2 and pdoc.indexed_at
        assert pdoc.uploaded_by == "tutor_1"
        doc = db.get(Document, pdoc.document_id)
        assert doc.status == "indexed" and doc.visibility == "class"
        assert db.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == doc.id)) == 2
        md = db.scalars(select(ModuleDocument).where(ModuleDocument.document_id == doc.id)).one()
        assert md.module_id == pdoc.module_id and md.version == 1 and md.is_latest
        assert fakes.ingestor.calls[0]["module_id"] == str(pdoc.module_id)
        assert fakes.ingestor.calls[0]["visibility"] == "class"

    def test_upload_does_not_rename_course(self, client, db, api_key, fakes):
        client.post("/api/v1/courses/ingest", headers=auth(api_key),
                    json={"course_id": "c1", "user_id": "tutor_1", "title": "Python 101"})
        upload(client, api_key)
        assert db.scalars(select(Module)).one().title == "Python 101"

    @pytest.mark.parametrize("filename,ctype", [
        ("photo.jpg", "image/jpeg"), ("script.exe", "application/octet-stream"), ("noext", "text/plain"),
    ])
    def test_invalid_type_400(self, client, api_key, fakes, filename, ctype):
        r = upload(client, api_key, filename=filename, content_type=ctype)
        assert r.status_code == 400
        assert "not supported" in r.json()["detail"]

    def test_md_accepted_even_with_generic_content_type(self, client, api_key, fakes):
        r = upload(client, api_key, filename="notes.md", content_type="application/octet-stream")
        assert r.status_code == 202

    def test_oversized_400(self, client, api_key, fakes, monkeypatch):
        monkeypatch.setattr(v1_documents, "MAX_FILE_SIZE", 10)
        r = upload(client, api_key, content=b"x" * 11)
        assert r.status_code == 400 and "too large" in r.json()["detail"]
        assert fakes.uploads == []

    def test_empty_400(self, client, api_key, fakes):
        assert upload(client, api_key, content=b"").status_code == 400

    def test_storage_not_configured_503(self, client, api_key, monkeypatch):
        monkeypatch.setattr(get_settings(), "cloudinary_cloud_name", "")
        assert upload(client, api_key).status_code == 503

    def test_index_failure_marks_failed(self, client, db, api_key, fakes):
        body = upload(client, api_key, filename="scan.pdf", content=b"FAIL").json()
        pdoc = db.get(PlatformDocument, uuid.UUID(body["id"]))
        assert pdoc.status == "failed" and pdoc.error_message == "unreadable"


# ── List ───────────────────────────────────────────────────────────────────

class TestList:
    def test_list_returns_course_documents(self, client, api_key, fakes):
        upload(client, api_key, filename="a.pdf")
        upload(client, api_key, filename="b.txt")
        upload(client, api_key, filename="other.pdf", course_id="c2")
        r = client.get("/api/v1/documents/list", params={"course_id": "c1"}, headers=auth(api_key))
        assert r.status_code == 200
        docs = r.json()
        assert isinstance(docs, list) and {d["filename"] for d in docs} == {"a.pdf", "b.txt"}
        assert all(d["status"] == "ready" for d in docs)

    def test_list_empty(self, client, api_key):
        r = client.get("/api/v1/documents/list", params={"course_id": "nothing"}, headers=auth(api_key))
        assert r.status_code == 200 and r.json() == []

    def test_list_isolated_per_api_key(self, client, db, api_key, fakes):
        upload(client, api_key)
        other, key_hash, prefix = generate_api_key()
        db.add(ApiKey(id=uuid.uuid4(), name="Other", key_hash=key_hash, key_prefix=prefix,
                      platform="moodle", owner_email="o@test.dev", is_active=True, request_count=0))
        db.commit()
        r = client.get("/api/v1/documents/list", params={"course_id": "c1"}, headers=auth(other))
        assert r.json() == []


# ── Delete ─────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_removes_everything(self, client, db, api_key, fakes):
        body = upload(client, api_key).json()
        pdoc = db.get(PlatformDocument, uuid.UUID(body["id"]))
        doc_id, public_id = pdoc.document_id, pdoc.cloudinary_public_id
        db.expire_all()

        r = client.delete(f"/api/v1/documents/{body['id']}", params={"course_id": "c1"}, headers=auth(api_key))
        assert r.status_code == 200 and r.json() == {"deleted": True, "document_id": body["id"]}

        assert db.get(PlatformDocument, uuid.UUID(body["id"])) is None
        assert db.get(Document, doc_id) is None
        assert db.scalar(select(func.count()).select_from(DocumentChunk)) == 0
        assert db.scalar(select(func.count()).select_from(ModuleDocument)) == 0
        assert fakes.deletes == [public_id]
        assert fakes.ts_deletes == [str(doc_id)]

    def test_wrong_course_404(self, client, api_key, fakes):
        body = upload(client, api_key).json()
        r = client.delete(f"/api/v1/documents/{body['id']}", params={"course_id": "c2"}, headers=auth(api_key))
        assert r.status_code == 404

    def test_unknown_id_404_and_bad_id_422(self, client, api_key):
        assert client.delete(f"/api/v1/documents/{uuid.uuid4()}", params={"course_id": "c1"},
                             headers=auth(api_key)).status_code == 404
        assert client.delete("/api/v1/documents/not-a-uuid", params={"course_id": "c1"},
                             headers=auth(api_key)).status_code == 422


# ── Replace ────────────────────────────────────────────────────────────────

class TestReplace:
    def test_replace_reindexes_new_version(self, client, db, api_key, fakes):
        body = upload(client, api_key, filename="week1.pdf").json()
        old  = db.get(PlatformDocument, uuid.UUID(body["id"]))
        old_public_id, doc_id = old.cloudinary_public_id, old.document_id
        db.expire_all()

        r = client.post(f"/api/v1/documents/{body['id']}/replace", headers=auth(api_key),
                        data={"course_id": "c1"},
                        files={"file": ("week1_v2.pdf", b"New A\nNew B\nNew C\n", "application/pdf")})
        assert r.status_code == 202, r.text
        assert r.json()["status"] == "indexing" and r.json()["filename"] == "week1_v2.pdf"

        pdoc = db.get(PlatformDocument, uuid.UUID(body["id"]))
        assert pdoc.status == "ready" and pdoc.chunk_count == 3 and pdoc.document_id == doc_id
        assert db.get(Document, doc_id).filename == "week1_v2.pdf"
        assert db.scalar(select(func.count()).select_from(DocumentChunk)) == 3
        md = db.scalars(select(ModuleDocument).where(ModuleDocument.document_id == doc_id)).one()
        assert md.version == 2
        # Old version's leftover search chunks removed, old file removed (different name)
        assert fakes.superseded == [(str(doc_id), 1)]
        assert fakes.deletes == [old_public_id]

    def test_replace_same_filename_keeps_overwritten_file(self, client, api_key, fakes):
        body = upload(client, api_key, filename="notes.pdf").json()
        client.post(f"/api/v1/documents/{body['id']}/replace", headers=auth(api_key),
                    data={"course_id": "c1"}, files={"file": ("notes.pdf", b"v2\n", "application/pdf")})
        assert fakes.deletes == []  # same public_id — overwritten in place, must not be deleted

    def test_replace_invalid_type_400(self, client, api_key, fakes):
        body = upload(client, api_key).json()
        r = client.post(f"/api/v1/documents/{body['id']}/replace", headers=auth(api_key),
                        data={"course_id": "c1"}, files={"file": ("x.png", b"img", "image/png")})
        assert r.status_code == 400

    def test_replace_wrong_course_404(self, client, api_key, fakes):
        body = upload(client, api_key).json()
        r = client.post(f"/api/v1/documents/{body['id']}/replace", headers=auth(api_key),
                        data={"course_id": "c2"}, files={"file": ("a.pdf", b"x", "application/pdf")})
        assert r.status_code == 404


# ── Auth + roles ───────────────────────────────────────────────────────────

class TestAuth:
    def test_unauthorized_401(self, client):
        assert client.get("/api/v1/documents/list", params={"course_id": "c1"}).status_code == 401
        r = client.post("/api/v1/documents/upload", data={"course_id": "c1", "user_id": "u"},
                        files={"file": ("a.pdf", b"x", "application/pdf")})
        assert r.status_code == 401

    def test_tutor_session_token_can_upload(self, client, api_key, fakes):
        token = session_token(client, api_key, role="tutor")
        r = upload(client, token, course_id=None, user_id=None)
        assert r.status_code == 202, r.text

    def test_student_session_token_cannot_manage(self, client, api_key, fakes):
        doc_id = upload(client, api_key).json()["id"]
        token  = session_token(client, api_key, user_id="stu_1", role="student")
        assert upload(client, token, course_id=None, user_id=None).status_code == 403
        assert client.delete(f"/api/v1/documents/{doc_id}", headers=auth(token)).status_code == 403
        r = client.post(f"/api/v1/documents/{doc_id}/replace", headers=auth(token),
                        files={"file": ("a.pdf", b"x", "application/pdf")})
        assert r.status_code == 403
        # ...but can list the course materials
        r = client.get("/api/v1/documents/list", headers=auth(token))
        assert r.status_code == 200 and len(r.json()) == 1

    def test_session_token_other_course_401(self, client, api_key, fakes):
        token = session_token(client, api_key, course_id="c1")
        r = client.get("/api/v1/documents/list", params={"course_id": "c2"}, headers=auth(token))
        assert r.status_code == 401


# ── Shared course module ───────────────────────────────────────────────────

INGEST = {"title": "Python 101", "sections": [{"title": "Basics", "lectures": [{"title": "Vars"}]}]}


class TestSharedCourseModule:
    @pytest.mark.asyncio
    async def test_concurrent_first_requests_share_one_module(self, client, db, api_key, fakes):
        """A panel's parallel first requests must not race into duplicate users/modules (was a 500)."""
        import asyncio
        import httpx
        body = {"course_id": "c1", "user_id": "tutor_1", **INGEST}
        transport = httpx.ASGITransport(app=app)  # one event loop, real concurrency
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            responses = await asyncio.gather(*[
                ac.post("/api/v1/courses/ingest", headers=auth(api_key), json=body) for _ in range(4)
            ])
        assert [r.status_code for r in responses if r.status_code >= 400] == []
        assert db.scalar(select(func.count()).select_from(Module)) == 1
        assert db.scalar(select(func.count()).select_from(PlatformCourse)) == 1

    def test_tutor_materials_reach_students(self, client, db, api_key, fakes):
        pdoc_id = upload(client, api_key, user_id="tutor_1").json()["id"]
        r = client.post("/api/v1/courses/ingest", headers=auth(api_key),
                        json={"course_id": "c1", "user_id": "stu_1", **INGEST})
        pdoc = db.get(PlatformDocument, uuid.UUID(pdoc_id))
        assert r.json()["module_id"] == str(pdoc.module_id)
        assert db.scalar(select(func.count()).select_from(Module)) == 1

    def test_identical_content_indexed_once(self, client, db, api_key, fakes):
        r1 = client.post("/api/v1/courses/ingest", headers=auth(api_key),
                         json={"course_id": "c1", "user_id": "stu_1", **INGEST})
        assert r1.json()["status"] == "indexing"
        r2 = client.post("/api/v1/courses/ingest", headers=auth(api_key),
                         json={"course_id": "c1", "user_id": "stu_2", **INGEST})
        assert r2.json()["status"] == "ready"
        assert len(fakes.ingestor.calls) == 1
        pcs = db.scalars(select(PlatformCourse)).all()
        assert {pc.status for pc in pcs} == {"ready"} and len({pc.module_id for pc in pcs}) == 1

    def test_changed_content_reindexes_then_retires_old(self, client, db, api_key, fakes):
        client.post("/api/v1/courses/ingest", headers=auth(api_key),
                    json={"course_id": "c1", "user_id": "stu_1", **INGEST})
        first_doc = db.scalars(select(Document)).one().id
        r = client.post("/api/v1/courses/ingest", headers=auth(api_key),
                        json={"course_id": "c1", "user_id": "stu_1", **INGEST, "description": "updated"})
        assert r.json()["status"] == "indexing"
        assert len(fakes.ingestor.calls) == 2
        assert fakes.ts_deletes == [str(first_doc)]
        latest = db.scalars(select(ModuleDocument).where(ModuleDocument.is_latest == True)).all()  # noqa: E712
        assert len(latest) == 1 and latest[0].version == 2

    def test_course_index_failure_marks_failed(self, client, db, api_key, fakes, monkeypatch):
        monkeypatch.setattr(v1_platform, "build_course_text", lambda req: "FAIL")
        client.post("/api/v1/courses/ingest", headers=auth(api_key),
                    json={"course_id": "c1", "user_id": "stu_1", **INGEST})
        pc = db.scalars(select(PlatformCourse)).one()
        assert pc.status == "failed" and pc.error_message == "unreadable"
