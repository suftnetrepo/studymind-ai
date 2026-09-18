"""
Unit tests — Pydantic schemas (no external dependencies needed).
Run: pytest tests/test_schemas.py -v
"""
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.db.schemas import (
    ChatRequest, ChatResponse, DocumentSchema,
    DocumentUploadResponse, SourceCitation,
    SessionCreate, StatsResponse,
)


class TestSourceCitation:
    def test_valid(self):
        c = SourceCitation(
            document_id=str(uuid.uuid4()),
            filename="report.pdf",
            chunk_index=3,
            content_snippet="Key finding in section 2.",
            relevance_score=0.87,
        )
        assert c.filename == "report.pdf"
        assert c.relevance_score == 0.87

    def test_snippet_over_300_rejected(self):
        """Pydantic rejects snippets over 300 chars — caller must truncate before creating."""
        with pytest.raises(ValidationError):
            SourceCitation(
                document_id="x",
                filename="f.txt",
                chunk_index=0,
                content_snippet="A" * 301,
                relevance_score=0.5,
            )

    def test_snippet_exactly_300_accepted(self):
        c = SourceCitation(
            document_id="x",
            filename="f.txt",
            chunk_index=0,
            content_snippet="A" * 300,
            relevance_score=0.5,
        )
        assert len(c.content_snippet) == 300

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            SourceCitation(
                document_id="x", filename="f", chunk_index=0,
                content_snippet="x", relevance_score=1.5,
            )
        with pytest.raises(ValidationError):
            SourceCitation(
                document_id="x", filename="f", chunk_index=0,
                content_snippet="x", relevance_score=-0.1,
            )


class TestChatRequest:
    def test_defaults(self):
        r = ChatRequest(message="hello")
        assert r.stream is False
        assert r.session_id is None

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_message_too_long(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 8193)

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="hi", top_k=0)
        with pytest.raises(ValidationError):
            ChatRequest(message="hi", top_k=21)
        r = ChatRequest(message="hi", top_k=10)
        assert r.top_k == 10


class TestSessionCreate:
    def test_default_title(self):
        s = SessionCreate()
        assert s.title == "New conversation"

    def test_custom_title(self):
        s = SessionCreate(title="My project docs")
        assert s.title == "My project docs"

    def test_title_too_long(self):
        with pytest.raises(ValidationError):
            SessionCreate(title="x" * 256)


class TestDocumentUploadResponse:
    def test_valid(self):
        r = DocumentUploadResponse(
            document_id=uuid.uuid4(),
            filename="spec.pdf",
            status="indexed",
            message="Indexed 42 chunks",
        )
        assert r.status == "indexed"


class TestStatsResponse:
    def test_with_latency(self):
        s = StatsResponse(
            total_sessions=5,
            total_messages=20,
            total_documents=3,
            total_chunks=120,
            avg_latency_ms=340.5,
        )
        assert s.avg_latency_ms == 340.5

    def test_without_latency(self):
        s = StatsResponse(
            total_sessions=0,
            total_messages=0,
            total_documents=0,
            total_chunks=0,
        )
        assert s.avg_latency_ms is None
