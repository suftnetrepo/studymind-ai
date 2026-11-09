"""Regression guard: AI endpoints must check module access before generation."""
from pathlib import Path


def test_ai_generation_handlers_authorize_module_first():
    source = Path("app/api/features.py").read_text()
    handlers = [
        ("generate_quiz_endpoint", "metered_generation"),
        ("generate_flashcards_endpoint", "metered_generation"),
        ("summarise_endpoint", "metered_generation"),
    ]
    for name, action in handlers:
        section = source.split(f"async def {name}(", 1)[1].split("\n@router.", 1)[0]
        assert "await _get_accessible_module(req.module_id, current_user, db)" in section
        assert section.index("await _get_accessible_module(") < section.index(action)


def test_chat_and_stream_authorize_before_session_creation():
    source = Path("app/api/chat.py").read_text()
    for name in ("chat", "chat_stream"):
        section = source.split(f"async def {name}(", 1)[1].split("\n@router.", 1)[0]
        assert "await _get_accessible_module(req.module_id, current_user, db)" in section
        assert section.index("await _get_accessible_module(") < section.index("_get_or_create_session(")
