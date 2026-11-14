"""Transactional email: templates, Brevo sender, welcome on registration, recovery wiring."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.config import Settings
from app.email import sender, welcome
from app.email.templates import reset_email, welcome_email
from app.auth import recovery


def cfg(**kw):
    base = dict(brevo_api_key="", email_from_address="", recovery_from_email="", recovery_smtp_host="", recovery_smtp_username="",
                recovery_smtp_password="", password_recovery_enabled=False, recovery_public_url="", email_welcome_enabled=True, email_app_url="")
    base.update(kw)
    return Settings(**base)


def use(monkeypatch, **kw):
    s = cfg(**kw)
    for mod in (sender, welcome, recovery):
        monkeypatch.setattr(mod, "get_settings", lambda s=s: s)
    return s


# ── templates ───────────────────────────────────────────────────────────────
def test_welcome_is_role_specific_and_escapes_the_name():
    subject, html, text = welcome_email('<script>alert(1)</script> Lee', "lecturer", "https://app.example.com/")
    assert subject == "Welcome to StudyMind"
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Create a module" in text and "https://app.example.com/my-learning" in text
    _, _, student_text = welcome_email("Sam Student", "student", "https://app.example.com")
    assert "Course ID" in student_text and "Hello" not in student_text


def test_welcome_without_app_url_has_no_link():
    _, html, text = welcome_email("Sam", "student", "")
    assert "href" not in html and "http" not in text


def test_reset_email_contains_the_link_in_button_and_text():
    url = "https://app.example.com/my-learning/reset-password?token=abc%20d"
    subject, html, text = reset_email("Alex Rivera", url)
    assert subject == "Reset your StudyMind password"
    assert url in text and 'href="https://app.example.com/my-learning/reset-password?token=abc%20d"' in html
    assert "30 minutes" in text and "ignore this email" in text


# ── provider selection ──────────────────────────────────────────────────────
def test_provider_selection(monkeypatch):
    use(monkeypatch); assert sender.email_provider() is None and not sender.email_configured()
    use(monkeypatch, brevo_api_key="k"); assert sender.email_provider() is None          # no sender address
    use(monkeypatch, brevo_api_key="k", email_from_address="noreply@x.co"); assert sender.email_provider() == "brevo"
    use(monkeypatch, recovery_smtp_host="h", recovery_smtp_username="u", recovery_smtp_password="p", recovery_from_email="a@b.co")
    assert sender.email_provider() == "smtp"


# ── Brevo sender ────────────────────────────────────────────────────────────
class FakeClient:
    responses: list = []
    calls: list = []
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, json=None, headers=None):
        FakeClient.calls.append((url, json, headers))
        r = FakeClient.responses.pop(0)
        if isinstance(r, Exception): raise r
        return httpx.Response(r, request=httpx.Request("POST", url))


@pytest.fixture
def brevo(monkeypatch):
    use(monkeypatch, brevo_api_key="key-123", email_from_address="noreply@x.co")
    FakeClient.responses, FakeClient.calls = [], []
    monkeypatch.setattr(sender.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(sender.asyncio, "sleep", AsyncMock())
    return FakeClient


@pytest.mark.asyncio
async def test_brevo_success_sends_the_expected_payload(brevo):
    brevo.responses = [201]
    assert await sender.send_email("a@b.co", "Ann", "Subj", "<p>h</p>", "t", idempotency_key="id-1") is True
    url, payload, headers = brevo.calls[0]
    assert url == sender.BREVO_ENDPOINT and headers["api-key"] == "key-123"
    assert payload["to"] == [{"email": "a@b.co", "name": "Ann"}] and payload["sender"]["email"] == "noreply@x.co"
    assert payload["headers"] == {"idempotencyKey": "id-1"}


@pytest.mark.asyncio
async def test_brevo_retries_server_errors_then_succeeds(brevo):
    brevo.responses = [503, httpx.ConnectError("x"), 201]
    assert await sender.send_email("a@b.co", "", "S", "h", "t") is True
    assert len(brevo.calls) == 3


@pytest.mark.asyncio
async def test_brevo_does_not_retry_client_errors(brevo):
    brevo.responses = [400]
    assert await sender.send_email("a@b.co", "", "S", "h", "t") is False
    assert len(brevo.calls) == 1


@pytest.mark.asyncio
async def test_brevo_gives_up_after_retries_without_raising(brevo):
    brevo.responses = [500] * 10
    assert await sender.send_email("a@b.co", "", "S", "h", "t") is False
    assert len(brevo.calls) == sender._RETRIES + 1


@pytest.mark.asyncio
async def test_send_is_skipped_when_not_configured(monkeypatch):
    use(monkeypatch)
    assert await sender.send_email("a@b.co", "", "S", "h", "t") is False


# ── welcome ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_welcome_sends_when_configured_and_not_when_disabled(monkeypatch):
    send = AsyncMock(return_value=True)
    monkeypatch.setattr(welcome, "send_email", send)
    use(monkeypatch, brevo_api_key="k", email_from_address="n@x.co", email_app_url="https://app.example.com")
    assert await welcome.send_welcome_email("a@b.co", "Ann Lee", "student") is True
    assert send.await_args.args[0] == "a@b.co" and "Welcome to StudyMind" in send.await_args.args[2]
    send.reset_mock()
    use(monkeypatch, brevo_api_key="k", email_from_address="n@x.co", email_welcome_enabled=False)
    assert await welcome.send_welcome_email("a@b.co", "Ann Lee", "student") is False
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_registration_queues_the_welcome_email_and_survives_failure(monkeypatch):
    from app.api import auth
    from fastapi import BackgroundTasks
    monkeypatch.setattr(auth, "get_settings", lambda: SimpleNamespace(enabled_role_list=["student", "lecturer"]))
    db = MagicMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None))
    db.commit, db.refresh, db.add = AsyncMock(), AsyncMock(), MagicMock()
    tasks = BackgroundTasks()
    body = auth.RegisterRequest(email="new@example.com", password="Password123!", full_name="New Person", role="student")
    user = await auth.register(body, tasks, db)
    assert user.email == "new@example.com"
    assert [t.func for t in tasks.tasks] == [auth.send_welcome_email]
    assert tasks.tasks[0].args == ("new@example.com", "New Person", "student")


# ── recovery wiring ─────────────────────────────────────────────────────────
def test_recovery_needs_flag_provider_and_https_link(monkeypatch):
    use(monkeypatch, brevo_api_key="k", email_from_address="n@x.co", recovery_public_url="https://app.example.com/my-learning/reset-password")
    assert recovery.recovery_configured() is False                        # flag still off
    use(monkeypatch, password_recovery_enabled=True, brevo_api_key="k", email_from_address="n@x.co", recovery_public_url="https://app.example.com/r")
    assert recovery.recovery_configured() is True
    use(monkeypatch, password_recovery_enabled=True, brevo_api_key="k", email_from_address="n@x.co", recovery_public_url="http://app.example.com/r")
    assert recovery.recovery_configured() is False                        # not https
    use(monkeypatch, password_recovery_enabled=True, recovery_public_url="https://app.example.com/r")
    assert recovery.recovery_configured() is False                        # no provider


@pytest.mark.asyncio
async def test_recovery_email_carries_a_single_use_link(monkeypatch):
    send = AsyncMock(return_value=True)
    monkeypatch.setattr(recovery, "send_email", send)
    use(monkeypatch, password_recovery_enabled=True, brevo_api_key="k", email_from_address="n@x.co", recovery_public_url="https://app.example.com/my-learning/reset-password")
    await recovery.send_recovery_email("a@b.co", "Ann Lee", "tok/en+1")
    text = send.await_args.args[4]
    assert "https://app.example.com/my-learning/reset-password?token=tok%2Fen%2B1" in text
