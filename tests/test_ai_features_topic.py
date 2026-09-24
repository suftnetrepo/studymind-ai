"""Topic scoping for quiz / flashcards / summary: the topic must reach the prompt, not just retrieval."""
import pytest

from app.agents import ai_features


@pytest.fixture
def captured(monkeypatch):
    seen = {"queries": [], "prompts": []}
    monkeypatch.setattr(ai_features, "_retrieve_context",
                        lambda query, **kw: (seen["queries"].append(query) or ("## Loops\\nfor x in y", 1)))

    def fake_llm_json(prompt):
        seen["prompts"].append(prompt)
        return [{"question": "Q", "options": ["a", "b"], "correct_answer": "a", "front": "F", "back": "B"}]
    monkeypatch.setattr(ai_features, "_call_llm_json", fake_llm_json)

    class LLM:
        def complete(self, prompt):
            seen["prompts"].append(prompt)
            return "## Summary"
    monkeypatch.setattr(ai_features, "get_llm", lambda: LLM())
    return seen


GENERATORS = [
    lambda topic: ai_features.generate_quiz(module_id="m", question_count=2, topic=topic),
    lambda topic: ai_features.generate_flashcards(module_id="m", max_cards=5, topic=topic),
    lambda topic: ai_features.generate_summary(module_id="m", scope="module", topic=topic),
]


@pytest.mark.parametrize("generate", GENERATORS)
def test_topic_scopes_prompt_and_retrieval(captured, generate):
    generate("Control Flow")
    assert captured["queries"] == ["Control Flow"]
    assert captured["prompts"][0].startswith('TOPIC FOCUS: Use ONLY the parts of the content below that are about "Control Flow"')


@pytest.mark.parametrize("generate", GENERATORS)
def test_no_topic_leaves_prompt_unchanged(captured, generate):
    generate(None)
    assert "TOPIC FOCUS" not in captured["prompts"][0]


def test_blank_topic_ignored():
    assert ai_features._with_topic_focus("P", "   ") == "P"
