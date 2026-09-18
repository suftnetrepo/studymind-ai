"""
Sprint 4 — Scoped RAG Q&A tests.
Tests cover: ScopedChatRequest schema, scope building logic,
scope indicator construction, no-content suggestions,
history formatting, citation extraction.
No live DB or API calls required.
"""
import pytest
from pydantic import ValidationError

from app.db.schemas import ScopedChatRequest, ScopeIndicator, SourceCitation
from app.agents.rag_pipeline import (
    _format_history,
    _extract_citations,
    _build_scope_from_request,
    _build_scope_indicator,
    _no_content_suggestions,
)
from app.retrieval.retriever import SearchScope
from llama_index.core.schema import TextNode, NodeWithScore


# ── ScopedChatRequest schema ───────────────────────────────────────────────

class TestScopedChatRequest:
    def test_defaults(self):
        r = ScopedChatRequest(message="What is a pointer?")
        assert r.scope_mode      == "everything"
        assert r.include_archived is False
        assert r.stream          is False
        assert r.module_id       is None

    def test_all_valid_scope_modes(self):
        for mode in ["everything", "class_only", "personal_only", "all_semesters"]:
            r = ScopedChatRequest(message="test", scope_mode=mode)
            assert r.scope_mode == mode

    def test_invalid_scope_mode_rejected(self):
        with pytest.raises(ValidationError):
            ScopedChatRequest(message="test", scope_mode="wrong_mode")

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ScopedChatRequest(message="")

    def test_message_too_long_rejected(self):
        with pytest.raises(ValidationError):
            ScopedChatRequest(message="x" * 8193)

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            ScopedChatRequest(message="test", top_k=0)
        with pytest.raises(ValidationError):
            ScopedChatRequest(message="test", top_k=21)
        r = ScopedChatRequest(message="test", top_k=10)
        assert r.top_k == 10

    def test_include_archived_flag(self):
        r = ScopedChatRequest(message="test", include_archived=True)
        assert r.include_archived is True

    def test_stream_flag(self):
        r = ScopedChatRequest(message="test", stream=True)
        assert r.stream is True


# ── _build_scope_from_request ──────────────────────────────────────────────

class TestBuildScopeFromRequest:
    def test_default_everything_scope(self):
        scope = _build_scope_from_request(student_id="stu-1")
        assert scope.include_class         is True
        assert scope.include_personal      is True
        assert scope.current_semester_only is True
        assert scope.latest_only           is True

    def test_class_only_mode(self):
        scope = _build_scope_from_request(student_id="stu-1", scope_mode="class_only")
        assert scope.include_class    is True
        assert scope.include_personal is False
        assert scope.personal_only    is False

    def test_personal_only_mode(self):
        scope = _build_scope_from_request(
            student_id="stu-1", scope_mode="personal_only"
        )
        assert scope.include_class    is False
        assert scope.include_personal is True
        assert scope.personal_only    is True
        assert scope.student_id       == "stu-1"

    def test_all_semesters_mode(self):
        scope = _build_scope_from_request(
            student_id="stu-1", scope_mode="all_semesters"
        )
        assert scope.current_semester_only is False
        assert scope.include_class         is True
        assert scope.include_personal      is True

    def test_include_archived_overrides_semester(self):
        scope = _build_scope_from_request(
            student_id="stu-1", include_archived=True
        )
        assert scope.current_semester_only is False

    def test_module_id_passed_through(self):
        scope = _build_scope_from_request(
            student_id="stu-1", module_id="mod-xyz"
        )
        assert scope.module_id == "mod-xyz"

    def test_course_code_passed_through(self):
        scope = _build_scope_from_request(
            student_id="stu-1", course_code="CSC109"
        )
        assert scope.course_code == "CSC109"

    def test_week_number_passed_through(self):
        scope = _build_scope_from_request(
            student_id="stu-1", week_number=3
        )
        assert scope.week_number == 3


# ── _build_scope_indicator ─────────────────────────────────────────────────

class TestBuildScopeIndicator:
    def _make_scope(self, **kwargs) -> SearchScope:
        return SearchScope(**kwargs)

    def test_course_code_in_description(self):
        scope = self._make_scope(course_code="CSC109")
        ind   = _build_scope_indicator(scope, "everything", 5, "CSC109", "2025-S2")
        assert "CSC109" in ind.description

    def test_week_number_in_description(self):
        scope = self._make_scope(week_number=3)
        ind   = _build_scope_indicator(scope, "everything", 3, None, None)
        assert "Week 3" in ind.description

    def test_semester_label_in_description(self):
        scope = self._make_scope()
        ind   = _build_scope_indicator(scope, "everything", 10, None, "Autumn 2025")
        assert "Autumn 2025" in ind.description

    def test_class_only_in_description(self):
        scope = self._make_scope()
        ind   = _build_scope_indicator(scope, "class_only", 4, None, None)
        assert "Class materials only" in ind.description

    def test_personal_only_in_description(self):
        scope = self._make_scope()
        ind   = _build_scope_indicator(scope, "personal_only", 2, None, None)
        assert "My notes only" in ind.description

    def test_document_count_set(self):
        scope = self._make_scope()
        ind   = _build_scope_indicator(scope, "everything", 7, None, None)
        assert ind.document_count == 7

    def test_scope_mode_preserved(self):
        scope = self._make_scope()
        ind   = _build_scope_indicator(scope, "class_only", 0, None, None)
        assert ind.scope_mode == "class_only"

    def test_empty_scope_fallback(self):
        scope = self._make_scope(current_semester_only=False)
        ind   = _build_scope_indicator(scope, "everything", 0, None, None)
        assert ind.description  # not empty


# ── _no_content_suggestions ────────────────────────────────────────────────

class TestNoContentSuggestions:
    def test_week_scoped_suggests_expand_weeks(self):
        scope       = SearchScope(week_number=3)
        suggestions = _no_content_suggestions(scope)
        assert any("week" in s.lower() for s in suggestions)

    def test_current_semester_only_suggests_archives(self):
        scope       = SearchScope(current_semester_only=True)
        suggestions = _no_content_suggestions(scope)
        assert any("semester" in s.lower() for s in suggestions)

    def test_personal_only_suggests_class(self):
        scope       = SearchScope(personal_only=True)
        suggestions = _no_content_suggestions(scope)
        assert any("class" in s.lower() for s in suggestions)

    def test_class_only_suggests_personal(self):
        scope       = SearchScope(include_class=True, include_personal=False)
        suggestions = _no_content_suggestions(scope)
        assert any("personal" in s.lower() or "note" in s.lower() for s in suggestions)

    def test_fallback_suggestion_always_present(self):
        scope       = SearchScope()
        suggestions = _no_content_suggestions(scope)
        assert len(suggestions) >= 1


# ── _format_history ────────────────────────────────────────────────────────

class TestFormatHistory:
    def test_empty_history(self):
        assert _format_history([]) == "(none)"

    def test_single_turn(self):
        history = [{"role": "user", "content": "Hello"}]
        result  = _format_history(history)
        assert "User: Hello" in result

    def test_truncates_to_10_turns(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(15)]
        result  = _format_history(history)
        # Only last 10
        assert "msg 5" in result
        assert "msg 14" in result
        assert "msg 0" not in result

    def test_both_roles_formatted(self):
        history = [
            {"role": "user",      "content": "What is X?"},
            {"role": "assistant", "content": "X is Y."},
        ]
        result = _format_history(history)
        assert "User: What is X?" in result
        assert "Assistant: X is Y." in result


# ── _extract_citations ─────────────────────────────────────────────────────

def _make_node(doc_id, filename, chunk_index, text, score):
    node = TextNode(
        text=text,
        metadata={
            "document_id": doc_id,
            "filename":    filename,
            "chunk_index": chunk_index,
        },
    )
    return NodeWithScore(node=node, score=score)


class TestExtractCitations:
    def test_empty_nodes(self):
        assert _extract_citations([]) == []

    def test_single_citation(self):
        nodes  = [_make_node("doc-1", "notes.pdf", 0, "Pointers store addresses.", 0.9)]
        result = _extract_citations(nodes)
        assert len(result) == 1
        assert result[0].filename == "notes.pdf"
        assert result[0].relevance_score == 0.9

    def test_deduplication_by_doc_chunk(self):
        nodes = [
            _make_node("doc-1", "notes.pdf", 0, "text A", 0.9),
            _make_node("doc-1", "notes.pdf", 0, "text A", 0.9),   # duplicate
        ]
        result = _extract_citations(nodes)
        assert len(result) == 1

    def test_sorted_by_score_descending(self):
        nodes = [
            _make_node("doc-1", "a.pdf", 0, "low",  0.6),
            _make_node("doc-2", "b.pdf", 0, "high", 0.95),
            _make_node("doc-3", "c.pdf", 0, "mid",  0.8),
        ]
        result = _extract_citations(nodes)
        scores = [c.relevance_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_filters_low_scores(self):
        nodes = [
            _make_node("doc-1", "a.pdf", 0, "below", 0.3),
            _make_node("doc-2", "b.pdf", 0, "above", 0.9),
        ]
        result = _extract_citations(nodes, threshold=0.5)
        assert len(result) == 1
        assert result[0].filename == "b.pdf"

    def test_snippet_truncated_to_280(self):
        long_text = "word " * 200   # > 280 chars
        nodes     = [_make_node("doc-1", "x.pdf", 0, long_text, 0.8)]
        result    = _extract_citations(nodes)
        assert len(result[0].content_snippet) <= 300

    def test_different_chunks_same_doc_both_included(self):
        nodes = [
            _make_node("doc-1", "a.pdf", 0, "chunk 0", 0.9),
            _make_node("doc-1", "a.pdf", 1, "chunk 1", 0.85),
        ]
        result = _extract_citations(nodes)
        assert len(result) == 2
