"""Quiz answer-position balancing must spread the correct answer without ever changing which option is correct."""
import copy
import random
from collections import Counter

import pytest

from app.agents import ai_features
from app.agents.ai_features import balance_option_positions


def make(n=5, correct="b", k=4, explanation="The content states this."):
    qs = []
    for i in range(n):
        opts = [{"id": "abcdefgh"[j], "text": f"Q{i} option {j}"} for j in range(k)]
        qs.append({"position": i, "question": f"Question {i}?", "options": opts, "correct_answer": correct,
                   "explanation": explanation, "source_chunk": "src"})
    return qs


def correct_text(q):
    return next(o["text"] for o in q["options"] if o["id"] == q["correct_answer"])


@pytest.mark.parametrize("seed", range(200))
def test_correctness_and_content_are_preserved(seed):
    qs = make(n=7, correct="c")
    before = copy.deepcopy(qs)
    balance_option_positions(qs, random.Random(seed))
    for b, a in zip(before, qs):
        assert correct_text(a) == correct_text(b)                                    # same correct answer text
        assert sorted(o["text"] for o in a["options"]) == sorted(o["text"] for o in b["options"])   # same options
        assert [o["id"] for o in a["options"]] == list("abcd")                       # ids contiguous, in order
        assert a["correct_answer"] in {o["id"] for o in a["options"]}
        assert a["question"] == b["question"] and a["source_chunk"] == b["source_chunk"]


def test_positions_are_spread_evenly_within_a_quiz():
    for seed in range(100):
        qs = make(n=5, correct="b")
        balance_option_positions(qs, random.Random(seed))
        counts = Counter(q["correct_answer"] for q in qs)
        assert max(counts.values()) <= 2, counts                  # ceil(5 / 4); never "everything at b"
        assert len(counts) >= 3, counts


def test_positions_are_uniform_over_many_questions():
    counts = Counter()
    for seed in range(200):
        qs = make(n=8, correct="b")
        balance_option_positions(qs, random.Random(seed))
        counts.update(q["correct_answer"] for q in qs)
    total = sum(counts.values())
    for letter in "abcd":
        assert 0.22 <= counts[letter] / total <= 0.28, counts      # 25% each, small tolerance


def test_model_bias_toward_one_position_is_removed():
    # The live run produced b, c, b, b, b (never a): the output must not reproduce that skew.
    qs = make(n=20, correct="b")
    balance_option_positions(qs, random.Random(1))
    counts = Counter(q["correct_answer"] for q in qs)
    assert all(counts[x] == 5 for x in "abcd"), counts


@pytest.mark.parametrize("text", ["All of the above", "None of the above", "Both a and b", "A and C", "Neither of these options",
                                  "The option above", "see below"])
def test_order_dependent_questions_are_left_exactly_as_generated(text):
    qs = make(n=3, correct="d")
    qs[1]["options"][3]["text"] = text
    before = copy.deepcopy(qs[1])
    balance_option_positions(qs, random.Random(0))
    assert qs[1] == before


def test_unusable_questions_are_skipped_not_corrupted():
    bad = [
        {"question": "tf", "options": [{"id": "a", "text": "True"}, {"id": "b", "text": "False"}], "correct_answer": "a", "explanation": ""},
        {"question": "no options", "options": None, "correct_answer": "a", "explanation": ""},
        {"question": "unknown correct", "options": make(1)[0]["options"], "correct_answer": "z", "explanation": ""},
        {"question": "str options", "options": ["x", "y", "z"], "correct_answer": "a", "explanation": ""},
    ]
    before = copy.deepcopy(bad)
    balance_option_positions(bad, random.Random(0))
    assert bad == before
    balance_option_positions([])                                   # no crash on empty


def test_option_letter_references_in_explanations_follow_the_move():
    qs = make(n=1, correct="b", explanation="Option b is right because the notes say so; option a and option c are wrong.")
    qs[0]["options"][1]["text"] = "THE RIGHT ONE"
    for seed in range(50):
        q = copy.deepcopy(qs)
        balance_option_positions(q, random.Random(seed))
        right = q[0]["correct_answer"]
        assert f"Option {right} is right" in q[0]["explanation"]
        letters = {o["text"]: o["id"] for o in q[0]["options"]}
        assert letters["THE RIGHT ONE"] == right


def _fake_generation(monkeypatch, raw):
    prompts = []
    monkeypatch.setattr(ai_features, "_retrieve_context", lambda **kw: ("context", 1))
    monkeypatch.setattr(ai_features, "_call_llm_json", lambda prompt: (prompts.append(prompt), copy.deepcopy(raw))[1])
    return prompts


def test_generate_quiz_returns_balanced_positions_for_a_model_that_always_answers_b(monkeypatch):
    raw = [{"question": f"Q{i}", "options": [{"id": x, "text": f"{x.upper()}-text-{i}"} for x in "abcd"],
            "correct_answer": "b", "explanation": f"Because B-text-{i} is in the notes.", "source_chunk": "s"} for i in range(8)]
    prompts = _fake_generation(monkeypatch, raw)
    qs = ai_features.generate_quiz(module_id="m", student_id="s", question_count=8, question_type="mcq")
    assert len(prompts) == 1                                       # balancing adds no model call
    counts = Counter(q["correct_answer"] for q in qs)
    assert all(counts[x] == 2 for x in "abcd"), counts
    for i, q in enumerate(qs):                                     # the correct option is still the one the model chose
        assert correct_text(q) == f"B-text-{i}"


def test_generate_quiz_leaves_true_false_untouched(monkeypatch):
    raw = [{"question": f"Q{i}", "options": [{"id": "a", "text": "True"}, {"id": "b", "text": "False"}],
            "correct_answer": "a", "explanation": "e", "source_chunk": "s"} for i in range(6)]
    _fake_generation(monkeypatch, raw)
    qs = ai_features.generate_quiz(module_id="m", student_id="s", question_count=6, question_type="true_false")
    assert all(q["options"] == [{"id": "a", "text": "True"}, {"id": "b", "text": "False"}] and q["correct_answer"] == "a" for q in qs)


def test_summary_prompt_separates_grounded_content_from_general_advice():
    p = ai_features.SUMMARY_PROMPT
    assert "Exam Tips" not in p and "likely exam topics" not in p
    grounded, _, advice = p.partition("## General study advice (not from your documents)")
    assert advice, "advice section heading missing"
    assert "Use ONLY the content below" in grounded
    for h in ("## Key Concepts", "## Main Arguments / Explanations", "## Important Definitions"):
        assert h in grounded
    assert "no subject" in advice and "no predictions" in advice
    assert p.index("## Important Definitions") < p.index("## General study advice")
