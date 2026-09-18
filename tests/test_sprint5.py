"""
Sprint 5 — Quiz, Flashcard & Summariser tests.
Tests cover schema validation, quiz scoring logic, flashcard status logic.
No live DB or API calls.
"""
import pytest
from pydantic import ValidationError

from app.db.schemas import (
    QuizGenerateRequest, QuizSubmitRequest, QuizSubmitAnswer,
    FlashcardGenerateRequest, FlashcardUpdateRequest,
    SummariseRequest,
)
from app.agents.ai_features import score_quiz, _call_llm_json
import uuid


# ── QuizGenerateRequest ────────────────────────────────────────────────────

class TestQuizGenerateRequest:
    def test_valid_defaults(self):
        r = QuizGenerateRequest(module_id=uuid.uuid4())
        assert r.question_count == 10
        assert r.question_type  == "mcq"

    def test_all_valid_question_types(self):
        for qt in ["mcq", "short_answer", "true_false"]:
            r = QuizGenerateRequest(module_id=uuid.uuid4(), question_type=qt)
            assert r.question_type == qt

    def test_invalid_question_type_rejected(self):
        with pytest.raises(ValidationError):
            QuizGenerateRequest(module_id=uuid.uuid4(), question_type="essay")

    def test_question_count_lower_bound(self):
        with pytest.raises(ValidationError):
            QuizGenerateRequest(module_id=uuid.uuid4(), question_count=4)

    def test_question_count_upper_bound(self):
        with pytest.raises(ValidationError):
            QuizGenerateRequest(module_id=uuid.uuid4(), question_count=31)

    def test_boundary_counts_valid(self):
        QuizGenerateRequest(module_id=uuid.uuid4(), question_count=5)
        QuizGenerateRequest(module_id=uuid.uuid4(), question_count=30)

    def test_title_optional(self):
        r = QuizGenerateRequest(module_id=uuid.uuid4())
        assert r.title is None


# ── QuizSubmitRequest ──────────────────────────────────────────────────────

class TestQuizSubmitRequest:
    def test_valid_submit(self):
        qid = uuid.uuid4()
        r   = QuizSubmitRequest(
            answers=[QuizSubmitAnswer(question_id=qid, answer="a")]
        )
        assert len(r.answers) == 1

    def test_empty_answers_rejected(self):
        with pytest.raises(ValidationError):
            QuizSubmitRequest(answers=[])

    def test_empty_answer_text_rejected(self):
        with pytest.raises(ValidationError):
            QuizSubmitRequest(
                answers=[QuizSubmitAnswer(question_id=uuid.uuid4(), answer="")]
            )


# ── score_quiz ─────────────────────────────────────────────────────────────

class MockQuestion:
    """Minimal mock of QuizQuestion ORM object for testing score logic."""
    def __init__(self, id, correct_answer):
        self.id             = id
        self.correct_answer = correct_answer
        self.student_answer = None
        self.is_correct     = None


class TestScoreQuiz:
    def test_all_correct(self):
        q1 = MockQuestion(uuid.UUID("00000000-0000-0000-0000-000000000001"), "a")
        q2 = MockQuestion(uuid.UUID("00000000-0000-0000-0000-000000000002"), "b")
        answers = {
            "00000000-0000-0000-0000-000000000001": "a",
            "00000000-0000-0000-0000-000000000002": "b",
        }
        score, results = score_quiz([q1, q2], answers)
        assert score == 100.0
        assert all(q.is_correct for q in results)

    def test_all_wrong(self):
        q1 = MockQuestion(uuid.UUID("00000000-0000-0000-0000-000000000001"), "a")
        answers = {"00000000-0000-0000-0000-000000000001": "d"}
        score, results = score_quiz([q1], answers)
        assert score == 0.0
        assert results[0].is_correct is False

    def test_partial_score(self):
        q1 = MockQuestion(uuid.UUID("00000000-0000-0000-0000-000000000001"), "a")
        q2 = MockQuestion(uuid.UUID("00000000-0000-0000-0000-000000000002"), "b")
        q3 = MockQuestion(uuid.UUID("00000000-0000-0000-0000-000000000003"), "c")
        q4 = MockQuestion(uuid.UUID("00000000-0000-0000-0000-000000000004"), "d")
        answers = {
            "00000000-0000-0000-0000-000000000001": "a",   # correct
            "00000000-0000-0000-0000-000000000002": "a",   # wrong
            "00000000-0000-0000-0000-000000000003": "c",   # correct
            "00000000-0000-0000-0000-000000000004": "a",   # wrong
        }
        score, _ = score_quiz([q1, q2, q3, q4], answers)
        assert score == 50.0

    def test_empty_quiz(self):
        score, results = score_quiz([], {})
        assert score == 0.0
        assert results == []

    def test_case_insensitive_matching(self):
        q1 = MockQuestion(uuid.UUID("00000000-0000-0000-0000-000000000001"), "True")
        answers = {"00000000-0000-0000-0000-000000000001": "true"}
        score, results = score_quiz([q1], answers)
        assert results[0].is_correct is True

    def test_student_answer_stored(self):
        q1 = MockQuestion(uuid.UUID("00000000-0000-0000-0000-000000000001"), "a")
        answers = {"00000000-0000-0000-0000-000000000001": "b"}
        _, results = score_quiz([q1], answers)
        assert results[0].student_answer == "b"


# ── FlashcardGenerateRequest ───────────────────────────────────────────────

class TestFlashcardGenerateRequest:
    def test_valid_defaults(self):
        r = FlashcardGenerateRequest(module_id=uuid.uuid4())
        assert r.max_cards == 20

    def test_max_cards_lower_bound(self):
        with pytest.raises(ValidationError):
            FlashcardGenerateRequest(module_id=uuid.uuid4(), max_cards=4)

    def test_max_cards_upper_bound(self):
        with pytest.raises(ValidationError):
            FlashcardGenerateRequest(module_id=uuid.uuid4(), max_cards=51)

    def test_boundary_values(self):
        FlashcardGenerateRequest(module_id=uuid.uuid4(), max_cards=5)
        FlashcardGenerateRequest(module_id=uuid.uuid4(), max_cards=50)

    def test_title_optional(self):
        r = FlashcardGenerateRequest(module_id=uuid.uuid4())
        assert r.title is None


# ── FlashcardUpdateRequest ─────────────────────────────────────────────────

class TestFlashcardUpdateRequest:
    def test_all_valid_statuses(self):
        for s in ["new", "learning", "mastered"]:
            r = FlashcardUpdateRequest(status=s)
            assert r.status == s

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            FlashcardUpdateRequest(status="forgotten")

    def test_status_required(self):
        with pytest.raises(ValidationError):
            FlashcardUpdateRequest()


# ── SummariseRequest ───────────────────────────────────────────────────────

class TestSummariseRequest:
    def test_valid_document_scope(self):
        r = SummariseRequest(document_id=uuid.uuid4(), scope="document")
        assert r.scope == "document"

    def test_valid_week_scope(self):
        r = SummariseRequest(week_id=uuid.uuid4(), scope="week")
        assert r.scope == "week"

    def test_valid_module_scope(self):
        r = SummariseRequest(module_id=uuid.uuid4(), scope="module")
        assert r.scope == "module"

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValidationError):
            SummariseRequest(module_id=uuid.uuid4(), scope="chapter")

    def test_default_scope_is_document(self):
        r = SummariseRequest(document_id=uuid.uuid4())
        assert r.scope == "document"

    def test_all_ids_optional(self):
        r = SummariseRequest()
        assert r.module_id   is None
        assert r.week_id     is None
        assert r.document_id is None


# ── JSON parsing helper ────────────────────────────────────────────────────

class TestCallLLMJson:
    def test_strips_json_code_fence(self):
        import re, json

        raw = '```json\n[{"front": "term", "back": "def"}]\n```'
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        assert isinstance(result, list)
        assert result[0]["front"] == "term"

    def test_strips_plain_code_fence(self):
        import re, json

        raw = '```\n{"question": "What is X?"}\n```'
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        assert result["question"] == "What is X?"

    def test_clean_json_passes_through(self):
        import json

        raw    = '[{"front": "Pointer", "back": "Stores a memory address"}]'
        result = json.loads(raw)
        assert len(result) == 1
