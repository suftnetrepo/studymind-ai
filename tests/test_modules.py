"""
Sprint 3 — Modules, Semesters, Weeks & SearchScope tests.
Tests cover: schema validation, SearchScope filter building,
scope shortcut parsing (/csc109/week3), version logic.
No live DB or API calls.
"""
import pytest
from pydantic import ValidationError

from app.db.schemas import (
    ModuleCreate, WeekCreate, SemesterCreate,
    EnrolRequest, ModuleDocumentSchema,
)
from app.retrieval.retriever import SearchScope


# ── ModuleCreate ───────────────────────────────────────────────────────────

class TestModuleCreate:
    def test_valid_class_module(self):
        m = ModuleCreate(
            title="Introduction to Programming",
            course_code="CSC109",
            access_type="class",
            status="active",
        )
        assert m.course_code == "CSC109"
        assert m.access_type == "class"

    def test_valid_personal_course(self):
        m = ModuleCreate(title="My ACCA Study", access_type="personal")
        assert m.status == "active"

    def test_all_valid_access_types(self):
        for t in ["personal", "class", "institution"]:
            m = ModuleCreate(title="Test", access_type=t)
            assert m.access_type == t

    def test_invalid_access_type_rejected(self):
        with pytest.raises(ValidationError):
            ModuleCreate(title="Test", access_type="public")

    def test_all_valid_statuses(self):
        for s in ["active", "archived", "draft"]:
            m = ModuleCreate(title="Test", status=s)
            assert m.status == s

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            ModuleCreate(title="Test", status="deleted")

    def test_title_required(self):
        with pytest.raises(ValidationError):
            ModuleCreate(title="")

    def test_course_code_optional(self):
        m = ModuleCreate(title="My Course")
        assert m.course_code is None

    def test_description_optional(self):
        m = ModuleCreate(title="Test")
        assert m.description is None


# ── WeekCreate ─────────────────────────────────────────────────────────────

class TestWeekCreate:
    def test_valid_week(self):
        w = WeekCreate(week_number=3, title="Binary Trees")
        assert w.week_number == 3

    def test_week_number_lower_bound(self):
        with pytest.raises(ValidationError):
            WeekCreate(week_number=0, title="Invalid")

    def test_week_number_upper_bound(self):
        with pytest.raises(ValidationError):
            WeekCreate(week_number=53, title="Invalid")

    def test_boundary_weeks_valid(self):
        WeekCreate(week_number=1, title="First")
        WeekCreate(week_number=52, title="Last")

    def test_title_too_short_rejected(self):
        with pytest.raises(ValidationError):
            WeekCreate(week_number=1, title="X")

    def test_description_optional(self):
        w = WeekCreate(week_number=1, title="Intro")
        assert w.description is None


# ── SemesterCreate ─────────────────────────────────────────────────────────

class TestSemesterCreate:
    def test_valid_semester(self):
        s = SemesterCreate(label="2025 Semester 1")
        assert s.is_current is True

    def test_label_too_short_rejected(self):
        with pytest.raises(ValidationError):
            SemesterCreate(label="X")

    def test_label_too_long_rejected(self):
        with pytest.raises(ValidationError):
            SemesterCreate(label="X" * 65)

    def test_is_current_defaults_true(self):
        s = SemesterCreate(label="Autumn 2025")
        assert s.is_current is True

    def test_dates_optional(self):
        s = SemesterCreate(label="2025-S1")
        assert s.start_date is None
        assert s.end_date is None


# ── EnrolRequest ───────────────────────────────────────────────────────────

class TestEnrolRequest:
    def test_valid_code(self):
        r = EnrolRequest(enrolment_code="abc123")
        assert r.enrolment_code == "ABC123"   # uppercased

    def test_code_uppercased(self):
        r = EnrolRequest(enrolment_code="lowercase")
        assert r.enrolment_code == "LOWERCASE"

    def test_both_optional(self):
        r = EnrolRequest()
        assert r.enrolment_code is None
        assert r.module_id is None


# ── SearchScope ────────────────────────────────────────────────────────────

class TestSearchScope:
    def test_default_scope_has_latest_and_current_semester(self):
        scope  = SearchScope()
        filter_ = scope.build_filter()
        assert "is_latest:=true" in filter_
        assert "is_current_semester:=true" in filter_

    def test_module_id_filter(self):
        scope  = SearchScope(module_id="mod-123", current_semester_only=False)
        filter_ = scope.build_filter()
        assert "module_id:=mod-123" in filter_

    def test_course_code_filter(self):
        scope  = SearchScope(course_code="CSC109", current_semester_only=False)
        filter_ = scope.build_filter()
        assert "course_code:=CSC109" in filter_

    def test_course_code_takes_precedence_over_module_id(self):
        scope  = SearchScope(
            course_code="CSC109",
            module_id="mod-123",
            current_semester_only=False,
        )
        filter_ = scope.build_filter()
        assert "course_code:=CSC109" in filter_
        assert "module_id:=mod-123" not in filter_

    def test_week_number_filter(self):
        scope  = SearchScope(week_number=3, current_semester_only=False)
        filter_ = scope.build_filter()
        assert "week_number:=3" in filter_

    def test_personal_only_requires_student_id(self):
        scope  = SearchScope(
            personal_only=True,
            student_id="student-uuid",
            current_semester_only=False,
        )
        filter_ = scope.build_filter()
        assert "visibility:=personal" in filter_
        assert "owner_id:=student-uuid" in filter_

    def test_class_and_personal_combined(self):
        scope  = SearchScope(
            student_id="stu-123",
            include_class=True,
            include_personal=True,
            current_semester_only=False,
        )
        filter_ = scope.build_filter()
        assert "visibility:=class" in filter_
        assert "owner_id:=stu-123" in filter_

    def test_class_only(self):
        scope  = SearchScope(
            include_class=True,
            include_personal=False,
            current_semester_only=False,
        )
        filter_ = scope.build_filter()
        assert "visibility:=class" in filter_
        assert "personal" not in filter_

    def test_multiple_module_ids(self):
        scope  = SearchScope(
            module_ids=["mod-1", "mod-2", "mod-3"],
            current_semester_only=False,
        )
        filter_ = scope.build_filter()
        assert "module_id:[mod-1,mod-2,mod-3]" in filter_

    def test_semester_id_overrides_current(self):
        scope  = SearchScope(
            semester_id="sem-abc",
            current_semester_only=True,
        )
        filter_ = scope.build_filter()
        assert "semester_id:=sem-abc" in filter_
        assert "is_current_semester" not in filter_


# ── SearchScope.from_shortcut ──────────────────────────────────────────────

class TestSearchScopeShortcut:
    def test_course_code_shortcut(self):
        scope = SearchScope.from_shortcut("/csc109", "stu-123")
        assert scope.course_code == "CSC109"
        assert scope.week_number is None

    def test_course_week_shortcut(self):
        scope = SearchScope.from_shortcut("/csc109/week3", "stu-123")
        assert scope.course_code == "CSC109"
        assert scope.week_number == 3

    def test_uppercase_course_code(self):
        scope = SearchScope.from_shortcut("/MTH201", "stu-xyz")
        assert scope.course_code == "MTH201"

    def test_week_only_numeric(self):
        scope = SearchScope.from_shortcut("/eng105/week12", "stu-1")
        assert scope.week_number == 12

    def test_invalid_week_number_ignored(self):
        scope = SearchScope.from_shortcut("/csc109/weekXYZ", "stu-1")
        assert scope.week_number is None

    def test_student_id_set(self):
        scope = SearchScope.from_shortcut("/csc109", "my-student-id")
        assert scope.student_id == "my-student-id"

    def test_shortcut_builds_valid_filter(self):
        scope   = SearchScope.from_shortcut("/csc109/week3", "stu-1")
        filter_ = scope.build_filter()
        assert "course_code:=CSC109" in filter_
        assert "week_number:=3" in filter_
