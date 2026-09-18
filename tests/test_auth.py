"""
Sprint 1 — Authentication tests.
Tests cover: schemas, password hashing, JWT creation/decode, role validation.
No live DB or API calls needed.
"""
import pytest
from datetime import datetime, timezone, timedelta
from jose import JWTError
from pydantic import ValidationError

from app.auth.security import (
    hash_password, verify_password,
    create_access_token, decode_access_token,
    generate_refresh_token, hash_refresh_token,
)
from app.db.schemas import RegisterRequest, LoginRequest, TokenResponse


# ── Password hashing ───────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = hash_password("SecurePass1")
        assert hashed != "SecurePass1"
        assert len(hashed) > 20

    def test_correct_password_verifies(self):
        hashed = hash_password("SecurePass1")
        assert verify_password("SecurePass1", hashed) is True

    def test_wrong_password_fails(self):
        hashed = hash_password("SecurePass1")
        assert verify_password("WrongPass1", hashed) is False

    def test_empty_password_fails(self):
        hashed = hash_password("SecurePass1")
        assert verify_password("", hashed) is False

    def test_two_hashes_of_same_password_differ(self):
        """bcrypt salts — same password must produce different hashes."""
        h1 = hash_password("SecurePass1")
        h2 = hash_password("SecurePass1")
        assert h1 != h2
        assert verify_password("SecurePass1", h1)
        assert verify_password("SecurePass1", h2)


# ── JWT ────────────────────────────────────────────────────────────────────

class TestJWT:
    def test_access_token_decodes_correctly(self):
        token   = create_access_token("user-123", "student", "a@b.com")
        payload = decode_access_token(token)
        assert payload["sub"]   == "user-123"
        assert payload["role"]  == "student"
        assert payload["email"] == "a@b.com"
        assert payload["type"]  == "access"

    def test_all_roles_encode_correctly(self):
        for role in ["admin", "lecturer", "student", "self_learner"]:
            token   = create_access_token("uid", role, "x@y.com")
            payload = decode_access_token(token)
            assert payload["role"] == role

    def test_tampered_token_raises(self):
        token = create_access_token("uid", "student", "a@b.com")
        bad   = token[:-5] + "XXXXX"
        with pytest.raises(JWTError):
            decode_access_token(bad)

    def test_empty_token_raises(self):
        with pytest.raises(JWTError):
            decode_access_token("")

    def test_refresh_token_rejected_as_access(self):
        """Refresh tokens must not be accepted as access tokens."""
        from app.config import get_settings
        from jose import jwt as jose_jwt
        s = get_settings()
        payload = {"sub": "uid", "type": "refresh", "exp": datetime.now(timezone.utc) + timedelta(days=7)}
        token = jose_jwt.encode(payload, s.jwt_secret_key, algorithm=s.jwt_algorithm)
        with pytest.raises(JWTError):
            decode_access_token(token)


# ── Refresh token ──────────────────────────────────────────────────────────

class TestRefreshToken:
    def test_raw_and_hash_differ(self):
        raw, hashed = generate_refresh_token()
        assert raw != hashed

    def test_hash_is_deterministic(self):
        raw, hashed = generate_refresh_token()
        assert hash_refresh_token(raw) == hashed

    def test_different_raws_produce_different_hashes(self):
        raw1, h1 = generate_refresh_token()
        raw2, h2 = generate_refresh_token()
        assert raw1 != raw2
        assert h1 != h2

    def test_raw_token_length(self):
        raw, _ = generate_refresh_token()
        assert len(raw) >= 64


# ── RegisterRequest schema ─────────────────────────────────────────────────

class TestRegisterRequest:
    def test_valid_student(self):
        r = RegisterRequest(
            email="student@uni.ac.uk",
            password="SecurePass1",
            full_name="Jane Smith",
            role="student",
        )
        assert r.role == "student"

    def test_all_valid_roles(self):
        for role in ["admin", "lecturer", "student", "self_learner"]:
            r = RegisterRequest(
                email=f"{role}@test.com",
                password="SecurePass1",
                full_name="Test User",
                role=role,
            )
            assert r.role == role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            RegisterRequest(
                email="x@x.com",
                password="SecurePass1",
                full_name="Test",
                role="superuser",
            )

    def test_password_too_short_rejected(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="x@x.com", password="Ab1", full_name="T", role="student")

    def test_password_no_uppercase_rejected(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="x@x.com", password="nouppercase1", full_name="T", role="student")

    def test_password_no_digit_rejected(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="x@x.com", password="NoDigitHere", full_name="T", role="student")

    def test_invalid_email_rejected(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="not-an-email", password="SecurePass1", full_name="T", role="student")
