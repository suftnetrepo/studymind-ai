"""Email bodies. Everything user-supplied is HTML-escaped; each message has a plain-text twin."""
from __future__ import annotations

from html import escape

BRAND = "#6d28d9"

_ROLE_INTRO = {
    "lecturer": (
        "Create a module, upload your lecture notes and slides, and share the Course ID with your students.",
        ["Create your first module from the Dashboard", "Upload notes, slides or Word documents", "Share the Course ID so students can join"],
    ),
    "student": (
        "Join your lecturer's module with its Course ID, then ask questions, practise with quizzes and flashcards, and get summaries from the material.",
        ["Paste the Course ID your lecturer shared", "Upload your own private notes", "Practise with quizzes and flashcards"],
    ),
}
_DEFAULT_INTRO = (
    "Upload your notes, then ask questions, practise with quizzes and flashcards, and get summaries from them.",
    ["Create or join a module", "Upload your notes", "Practise with quizzes and flashcards"],
)


def _first_name(full_name: str) -> str:
    parts = (full_name or "").strip().split()
    return parts[0] if parts else "there"


def _layout(title: str, body_html: str) -> str:
    return (
        '<!doctype html><html><body style="margin:0;background:#f4f3fb;font-family:Arial,Helvetica,sans-serif;color:#1c1b3a">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:28px 12px">'
        '<table role="presentation" width="560" cellpadding="0" cellspacing="0" style="max-width:560px;background:#ffffff;border-radius:18px;overflow:hidden">'
        f'<tr><td style="background:{BRAND};padding:22px 28px;color:#ffffff;font-size:22px;font-weight:bold">StudyMind</td></tr>'
        f'<tr><td style="padding:28px"><h1 style="margin:0 0 14px;font-size:22px">{escape(title)}</h1>{body_html}</td></tr>'
        '<tr><td style="padding:16px 28px;background:#faf9ff;color:#6b6a8a;font-size:12px">One account for StudyMind Web and Mobile.</td></tr>'
        '</table></td></tr></table></body></html>'
    )


def _button(url: str, label: str) -> str:
    return (f'<p style="margin:22px 0"><a href="{escape(url, quote=True)}" '
            f'style="background:{BRAND};color:#ffffff;text-decoration:none;padding:12px 22px;border-radius:10px;font-weight:bold;display:inline-block">{escape(label)}</a></p>')


def welcome_email(full_name: str, role: str, app_url: str) -> tuple[str, str, str]:
    """(subject, html, text)"""
    name = _first_name(full_name)
    intro, steps = _ROLE_INTRO.get(role, _DEFAULT_INTRO)
    url = (app_url or "").rstrip("/") + "/my-learning" if app_url else ""
    subject = "Welcome to StudyMind"
    items = "".join(f'<li style="margin:4px 0">{escape(s)}</li>' for s in steps)
    html = _layout(f"Welcome, {name}!", (
        f'<p style="line-height:1.6;margin:0 0 12px">Your StudyMind account is ready. {escape(intro)}</p>'
        f'<ul style="padding-left:20px;line-height:1.6;margin:0 0 6px">{items}</ul>'
        + (_button(url, "Open your Dashboard") if url else "")
        + '<p style="color:#6b6a8a;font-size:13px;line-height:1.5">Use the same email and password on the StudyMind mobile app.</p>'))
    text = (f"Welcome, {name}!\n\nYour StudyMind account is ready. {intro}\n\n"
            + "\n".join(f"- {s}" for s in steps)
            + (f"\n\nOpen your Dashboard: {url}" if url else "")
            + "\n\nUse the same email and password on the StudyMind mobile app.\n")
    return subject, html, text


def reset_email(full_name: str, reset_url: str, minutes: int = 30) -> tuple[str, str, str]:
    """(subject, html, text)"""
    name = _first_name(full_name)
    subject = "Reset your StudyMind password"
    html = _layout("Reset your password", (
        f'<p style="line-height:1.6;margin:0 0 12px">Hi {escape(name)}, we received a request to reset your StudyMind password. '
        f'The link below works once and expires in {minutes} minutes.</p>'
        + _button(reset_url, "Choose a new password")
        + '<p style="color:#6b6a8a;font-size:13px;line-height:1.5">If you did not ask for this, you can ignore this email. Your password stays the same.</p>'
        f'<p style="color:#6b6a8a;font-size:12px;word-break:break-all">Button not working? Copy this link: {escape(reset_url)}</p>'))
    text = (f"Hi {name},\n\nWe received a request to reset your StudyMind password. Use this link within {minutes} minutes "
            f"(it works once):\n\n{reset_url}\n\nIf you did not ask for this, ignore this email. Your password stays the same.\n")
    return subject, html, text
