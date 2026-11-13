from app.agents import ai_features
from app.agents.formatting import JSON_RULES, MARKDOWN_RULES
from app.agents.rag_pipeline import build_system_prompt


def test_chat_system_prompt_carries_the_markdown_math_rules():
    for level in ("simple", "normal", "expert"):
        p = build_system_prompt(level)
        assert MARKDOWN_RULES.strip() in p and "double dollar" in p and "Never use \\( \\)" in p


def test_json_rules_tell_the_model_to_double_backslashes():
    assert "written twice" in JSON_RULES and "\\\\frac" in JSON_RULES


def test_feature_prompts_include_the_rules(monkeypatch):
    seen = {}

    def fake_json(prompt):
        seen["prompt"] = prompt
        return [{"question": "q", "options": [{"id": "a", "text": "x"}], "correct_answer": "a", "explanation": "e", "source_chunk": "s"}]

    monkeypatch.setattr(ai_features, "_call_llm_json", fake_json)
    monkeypatch.setattr(ai_features, "_retrieve_context", lambda *a, **k: ("some content", 1))
    try:
        ai_features.generate_quiz(module_id="00000000-0000-4000-8000-000000000001", student_id="s", question_count=1)
    except Exception:
        pass
    assert "written twice" in seen.get("prompt", "")
