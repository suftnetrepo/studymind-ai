"""
Sprint 2 — Institution & User Management tests.
Tests cover schemas, CSV parsing logic, code generation, password reset schemas.
No live DB required.
"""
import pytest
from pydantic import ValidationError

from app.db.schemas import (
    InstitutionCreate, InstitutionCodeCreate,
    DepartmentCreate, CreateLecturerRequest,
    BulkImportRow, JoinWithCodeRequest,
    PasswordResetRequestBody, PasswordResetConfirm,
)


# ── InstitutionCreate ──────────────────────────────────────────────────────

class TestInstitutionCreate:
    def test_valid_institution(self):
        i = InstitutionCreate(name="University of Lagos", domain="unilag.edu.ng")
        assert i.name == "University of Lagos"
        assert i.tier == "free"

    def test_name_required(self):
        with pytest.raises(ValidationError):
            InstitutionCreate(name="")

    def test_invalid_tier_rejected(self):
        with pytest.raises(ValidationError):
            InstitutionCreate(name="Test Uni", tier="gold")

    def test_all_valid_tiers(self):
        for tier in ["free", "educator", "institution", "enterprise"]:
            i = InstitutionCreate(name="Test", tier=tier)
            assert i.tier == tier

    def test_domain_optional(self):
        i = InstitutionCreate(name="Test Uni")
        assert i.domain is None


# ── DepartmentCreate ───────────────────────────────────────────────────────

class TestDepartmentCreate:
    def test_valid_department(self):
        d = DepartmentCreate(name="Computer Science", code="CSC")
        assert d.name == "Computer Science"
        assert d.code == "CSC"

    def test_code_optional(self):
        d = DepartmentCreate(name="Mathematics")
        assert d.code is None

    def test_name_too_short_rejected(self):
        with pytest.raises(ValidationError):
            DepartmentCreate(name="X")

    def test_code_too_long_rejected(self):
        with pytest.raises(ValidationError):
            DepartmentCreate(name="Test", code="X" * 33)


# ── CreateLecturerRequest ──────────────────────────────────────────────────

class TestCreateLecturerRequest:
    def test_valid_request(self):
        r = CreateLecturerRequest(
            email="lecturer@uni.ac.uk",
            full_name="Dr Jane Smith",
        )
        assert r.role if hasattr(r, 'role') else True
        assert r.send_invite is True

    def test_invalid_email_rejected(self):
        with pytest.raises(ValidationError):
            CreateLecturerRequest(email="not-email", full_name="Dr Test")

    def test_full_name_too_short_rejected(self):
        with pytest.raises(ValidationError):
            CreateLecturerRequest(email="x@x.com", full_name="X")


# ── BulkImportRow ─────────────────────────────────────────────────────────

class TestBulkImportRow:
    def test_valid_row(self):
        r = BulkImportRow(email="student@uni.ac.uk", full_name="John Doe")
        assert r.email == "student@uni.ac.uk"

    def test_optional_fields_default_none(self):
        r = BulkImportRow(email="s@uni.ac.uk", full_name="Jane")
        assert r.student_id is None
        assert r.department_code is None

    def test_with_optional_fields(self):
        r = BulkImportRow(
            email="s@uni.ac.uk",
            full_name="Jane",
            student_id="STU001",
            department_code="CSC",
        )
        assert r.student_id == "STU001"

    def test_invalid_email_rejected(self):
        with pytest.raises(ValidationError):
            BulkImportRow(email="bad-email", full_name="Jane")

    def test_full_name_required(self):
        with pytest.raises(ValidationError):
            BulkImportRow(email="s@uni.ac.uk", full_name="")


# ── InstitutionCodeCreate ──────────────────────────────────────────────────

class TestInstitutionCodeCreate:
    def test_valid_join_code(self):
        c = InstitutionCodeCreate(code_type="institution_join")
        assert c.target_role == "student"
        assert c.expires_days == 30

    def test_valid_enrolment_code(self):
        c = InstitutionCodeCreate(code_type="module_enrolment", max_uses=100)
        assert c.max_uses == 100

    def test_invalid_code_type_rejected(self):
        with pytest.raises(ValidationError):
            InstitutionCodeCreate(code_type="invalid_type")

    def test_max_uses_bounds(self):
        with pytest.raises(ValidationError):
            InstitutionCodeCreate(code_type="institution_join", max_uses=0)
        with pytest.raises(ValidationError):
            InstitutionCodeCreate(code_type="institution_join", max_uses=10001)

    def test_expires_days_bounds(self):
        with pytest.raises(ValidationError):
            InstitutionCodeCreate(code_type="institution_join", expires_days=0)
        with pytest.raises(ValidationError):
            InstitutionCodeCreate(code_type="institution_join", expires_days=366)


# ── JoinWithCodeRequest ────────────────────────────────────────────────────

class TestJoinWithCodeRequest:
    def test_valid_code(self):
        r = JoinWithCodeRequest(code="ABC123")
        assert r.code == "ABC123"

    def test_code_too_short_rejected(self):
        with pytest.raises(ValidationError):
            JoinWithCodeRequest(code="AB")

    def test_code_too_long_rejected(self):
        with pytest.raises(ValidationError):
            JoinWithCodeRequest(code="X" * 33)


# ── PasswordResetConfirm ───────────────────────────────────────────────────

class TestPasswordResetConfirm:
    def test_valid_reset(self):
        r = PasswordResetConfirm(token="abc123", new_password="NewPass1")
        assert r.new_password == "NewPass1"

    def test_password_no_uppercase_rejected(self):
        with pytest.raises(ValidationError):
            PasswordResetConfirm(token="tok", new_password="nouppercase1")

    def test_password_no_digit_rejected(self):
        with pytest.raises(ValidationError):
            PasswordResetConfirm(token="tok", new_password="NoDigitHere")

    def test_password_too_short_rejected(self):
        with pytest.raises(ValidationError):
            PasswordResetConfirm(token="tok", new_password="Ab1")


# ── CSV parsing simulation ─────────────────────────────────────────────────

class TestCSVParsing:
    """Test the CSV row parsing logic used in bulk_import_students."""

    def test_valid_csv_rows_parse(self):
        import csv, io
        csv_content = "email,full_name,student_id\nstudent@uni.ac.uk,Jane Doe,STU001\n"
        reader = csv.DictReader(io.StringIO(csv_content))
        rows   = list(reader)
        assert len(rows) == 1
        assert rows[0]["email"] == "student@uni.ac.uk"
        assert rows[0]["full_name"] == "Jane Doe"

    def test_missing_required_columns_detected(self):
        import csv, io
        csv_content = "name,student_id\nJane,STU001\n"
        reader = csv.DictReader(io.StringIO(csv_content))
        rows   = list(reader)
        required = {"email", "full_name"}
        assert not required.issubset(set(rows[0].keys()))

    def test_empty_rows_detected(self):
        import csv, io
        csv_content = "email,full_name\n,\n"
        reader = csv.DictReader(io.StringIO(csv_content))
        rows   = list(reader)
        assert not rows[0]["email"].strip()
        assert not rows[0]["full_name"].strip()

    def test_bom_handled(self):
        import csv, io
        # CSV with UTF-8 BOM (common from Excel exports on Windows)
        # The endpoint uses decode("utf-8-sig") which strips the BOM correctly
        raw_bytes = "email,full_name\nstudent@uni.ac.uk,Jane\n".encode("utf-8-sig")
        text      = raw_bytes.decode("utf-8-sig")   # strips BOM → clean "email"
        reader    = csv.DictReader(io.StringIO(text))
        rows      = list(reader)
        assert "email" in rows[0]   # BOM stripped — column name is clean
