"""
AI Features — Sprint 5.
Quiz generation, flashcard generation, and lecture summarisation.
All use GPT-4o with structured JSON output via the RAG retriever for context.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from llama_index.core.schema import QueryBundle

from app.agents.llm_factory import get_embed_model, get_llm
from app.config import get_settings
from app.logging_config import get_logger
from app.retrieval.retriever import SearchScope, ScopedHybridRetriever
from app.retrieval.typesense_client import get_typesense_client

log = get_logger(__name__)


# ── Context retrieval helper ───────────────────────────────────────────────

def _retrieve_context(
    query: str,
    module_id: str | None   = None,
    week_id: str | None     = None,
    document_id: str | None = None,
    student_id: str         = "",
    top_k: int              = 20,
) -> tuple[str, int]:
    """
    Retrieve relevant chunks for a module/week/document.
    Returns (context_text, chunk_count).
    For summarisation we use a high top_k to get comprehensive coverage.
    """
    embed_model = get_embed_model()
    scope       = SearchScope(
        student_id=student_id,
        module_id=module_id,
        week_id=week_id,
        current_semester_only=False,  # features should work across semesters
        latest_only=True,
        include_class=True,
        include_personal=True,
    )

    retriever = ScopedHybridRetriever(
        embed_model=embed_model,
        scope=scope,
        top_k=top_k,
        score_threshold=0.0,  # for features, include all content not just top hits
    )

    query_bundle     = QueryBundle(query_str=query)
    nodes_with_score = retriever.retrieve(query_bundle)

    parts = []
    for nws in nodes_with_score:
        meta   = nws.node.metadata
        fname  = meta.get("filename", "doc")
        cidx   = meta.get("chunk_index", 0)
        parts.append(f"[{fname}, chunk {cidx}]\n{nws.node.text}")

    context = "\n\n".join(parts)
    return context, len(nodes_with_score)


def _with_topic_focus(prompt: str, topic: str | None) -> str:
    """
    Scope a generation prompt to one topic (e.g. a course section or lecture title).
    Retrieval alone can't do this — small courses return all their chunks for any query —
    so the model is told explicitly to stay on topic.
    """
    if not topic or not topic.strip():
        return prompt
    return (
        f'TOPIC FOCUS: Use ONLY the parts of the content below that are about "{topic.strip()}" '
        "(for example the section or lecture with that title and its material). Ignore content "
        "on other topics. If there is little material on this topic, produce fewer items rather "
        "than drifting off-topic.\n\n"
        + prompt
    )


# Difficulty guidance per feature. "normal" leaves the prompt unchanged.
_COMPLEXITY_GUIDANCE: dict[str, dict[str, str]] = {
    "quiz": {
        "simple": "DIFFICULTY: SIMPLE — straightforward recall questions in plain, everyday language. "
                  "Avoid jargon; keep options short; wrong options should be clearly wrong.",
        "expert": "DIFFICULTY: EXPERT — challenging questions that test deeper understanding: application, "
                  "comparison, edge cases and why things work. Use precise technical terminology and "
                  "make the wrong options plausible.",
    },
    "flashcards": {
        "simple": "LEVEL: SIMPLE — explain each back side in plain, everyday language for a beginner, "
                  "one or two short sentences, with an everyday analogy where it helps. Avoid jargon.",
        "expert": "LEVEL: EXPERT — make each back side precise and technical: exact definitions, nuances, "
                  "edge cases and a short example (e.g. code) where relevant.",
    },
    "summary": {
        "simple": "LEVEL: SIMPLE — write for a beginner: plain language, short sentences, everyday "
                  "analogies, and explain any technical term the first time it appears.",
        "expert": "LEVEL: EXPERT — write for an advanced student: technical depth, precise terminology, "
                  "nuances, trade-offs and edge cases.",
    },
}


def _with_complexity(prompt: str, feature: str, complexity: str | None) -> str:
    guidance = _COMPLEXITY_GUIDANCE.get(feature, {}).get(complexity or "normal")
    return f"{guidance}\n\n{prompt}" if guidance else prompt


def _call_llm_json(prompt: str) -> Any:
    """
    Call GPT-4o and parse the JSON response.
    Strips markdown code fences if present.
    """
    llm      = get_llm()
    response = llm.complete(prompt)
    raw      = str(response).strip()

    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("llm_json_parse_error", error=str(e), raw=raw[:200])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e


# ═══════════════════════════════════════════════════════════════════════════
# Quiz Generator
# ═══════════════════════════════════════════════════════════════════════════

QUIZ_PROMPT = """\
You are an expert educator creating a quiz to test student understanding.

Using ONLY the content provided below, generate exactly {count} {q_type} questions.

Rules:
- Base every question strictly on the provided content — no outside knowledge.
- Each question must have a clear, unambiguous correct answer.
- Explanations must cite which part of the content the answer comes from.
- For MCQ: provide exactly 4 options labelled a, b, c, d.
- For true_false: options are only ["True", "False"].
- For short_answer: no options needed.

Return ONLY a valid JSON array (no markdown, no preamble):
[
  {{
    "question": "...",
    "options": [{{"id": "a", "text": "..."}}, ...],
    "correct_answer": "a",
    "explanation": "...",
    "source_chunk": "brief quote from content that supports the answer"
  }},
  ...
]

Content:
{context}
"""


def generate_quiz(
    module_id: str | None   = None,
    week_id: str | None     = None,
    document_id: str | None = None,
    student_id: str         = "",
    question_count: int     = 10,
    question_type: str      = "mcq",
    title: str              = "Quiz",
    topic: str | None       = None,
    complexity: str         = "normal",
) -> list[dict]:
    """
    QUIZ-01/02/03: Generate quiz questions from module/week/document content.
    Returns list of question dicts ready to be stored as QuizQuestion rows.
    """
    query = topic if topic else f"key concepts, definitions, important facts for {question_type} questions"
    context, chunk_count = _retrieve_context(
        query=query,
        module_id=module_id,
        week_id=week_id,
        document_id=document_id,
        student_id=student_id,
        top_k=min(question_count * 3, 30),
    )

    if not context:
        raise ValueError("No content found to generate quiz from. Upload materials first.")

    q_type_label = {
        "mcq":          "multiple-choice (MCQ)",
        "short_answer": "short answer",
        "true_false":   "true/false",
    }.get(question_type, "multiple-choice")

    prompt = _with_topic_focus(QUIZ_PROMPT.format(
        count=question_count,
        q_type=q_type_label,
        context=context[:12000],  # stay within context window
    ), topic)
    prompt = _with_complexity(prompt, "quiz", complexity)

    questions_raw = _call_llm_json(prompt)

    if not isinstance(questions_raw, list):
        raise ValueError("LLM did not return a list of questions")

    # Normalise and validate
    questions = []
    for i, q in enumerate(questions_raw[:question_count]):
        options, correct = _normalise_options(q.get("options"), str(q.get("correct_answer", "")), question_type)

        questions.append({
            "position":      i,
            "question":      str(q.get("question", "")).strip(),
            "options":       options,
            "correct_answer": correct,
            "explanation":   str(q.get("explanation", "")).strip(),
            "source_chunk":  str(q.get("source_chunk", ""))[:500],
        })

    log.info(
        "quiz_generated",
        question_count=len(questions),
        question_type=question_type,
        chunks_used=chunk_count,
    )
    return questions


def _normalise_options(raw_options, raw_correct: str, question_type: str):
    """
    Always return [{id, text}] options and a correct_answer that is one of those ids, whatever shape
    the model produced (plain strings, "A"/"True" answers, missing options for true/false...).
    """
    ids = "abcdefgh"
    if question_type == "true_false" or not raw_options:
        if question_type == "true_false":
            options = [{"id": "a", "text": "True"}, {"id": "b", "text": "False"}]
        else:
            return raw_options, raw_correct.strip().lower()
    else:
        options = []
        for idx, o in enumerate(raw_options):
            if isinstance(o, dict):
                options.append({"id": str(o.get("id", ids[idx])).strip().lower(), "text": str(o.get("text", "")).strip()})
            else:
                options.append({"id": ids[idx], "text": str(o).strip()})

    c = raw_correct.strip().lower()
    valid_ids = {o["id"] for o in options}
    if c not in valid_ids:
        by_text = next((o["id"] for o in options if o["text"].strip().lower() == c), None)
        c = by_text or (c[:1] if c[:1] in valid_ids else options[0]["id"])
    return options, c


def score_quiz(
    questions: list,    # QuizQuestion ORM objects
    answers: dict,      # {question_id_str: student_answer_str}
) -> tuple[float, list]:
    """
    QUIZ-05: Score a submitted quiz.
    Returns (score_percent, updated_questions_with_is_correct).
    """
    correct = 0
    results = []

    for q in questions:
        student_ans = answers.get(str(q.id), "").strip().lower()
        correct_ans = q.correct_answer.strip().lower()

        # For MCQ compare option id; for others compare full text
        is_correct = bool(student_ans) and student_ans == correct_ans

        q.student_answer = answers.get(str(q.id), "")
        q.is_correct     = is_correct
        if is_correct:
            correct += 1
        results.append(q)

    score = round((correct / len(questions)) * 100, 1) if questions else 0.0
    log.info("quiz_scored", correct=correct, total=len(questions), score=score)
    return score, results


# ═══════════════════════════════════════════════════════════════════════════
# Flashcard Generator
# ═══════════════════════════════════════════════════════════════════════════

FLASHCARD_PROMPT = """\
You are an expert educator creating flashcards to help a student memorise key concepts.

Using ONLY the content provided below, extract exactly {count} key term-definition pairs
as flashcards. Focus on: definitions, key concepts, important facts, formulas, names.

Rules:
- Base every card strictly on the provided content.
- Front (term): concise — typically 2-8 words.
- Back (definition): clear and complete — 1-3 sentences.
- source_chunk: a brief phrase from the content proving this fact is there.

Return ONLY a valid JSON array (no markdown, no preamble):
[
  {{
    "front": "term or concept",
    "back": "clear definition or explanation",
    "source_chunk": "brief quote from content"
  }},
  ...
]

Content:
{context}
"""


def generate_flashcards(
    module_id: str | None   = None,
    week_id: str | None     = None,
    document_id: str | None = None,
    student_id: str         = "",
    max_cards: int          = 20,
    topic: str | None       = None,
    complexity: str         = "normal",
) -> list[dict]:
    """
    FLASH-01/02: Generate flashcard deck from module/week/document content.
    Returns list of card dicts ready to be stored as Flashcard rows.
    """
    query = topic if topic else "key terms, definitions, important concepts and facts"
    context, chunk_count = _retrieve_context(
        query=query,
        module_id=module_id,
        week_id=week_id,
        document_id=document_id,
        student_id=student_id,
        top_k=min(max_cards * 2, 40),
    )

    if not context:
        raise ValueError("No content found to generate flashcards from. Upload materials first.")

    prompt = _with_topic_focus(FLASHCARD_PROMPT.format(
        count=max_cards,
        context=context[:12000],
    ), topic)
    prompt = _with_complexity(prompt, "flashcards", complexity)

    cards_raw = _call_llm_json(prompt)

    if not isinstance(cards_raw, list):
        raise ValueError("LLM did not return a list of cards")

    cards = []
    for i, c in enumerate(cards_raw[:max_cards]):
        front = str(c.get("front", "")).strip()
        back  = str(c.get("back", "")).strip()
        if not front or not back:
            continue
        cards.append({
            "position":    i,
            "front":       front,
            "back":        back,
            "source_chunk": str(c.get("source_chunk", ""))[:300],
            "status":      "new",
        })

    log.info(
        "flashcards_generated",
        card_count=len(cards),
        chunks_used=chunk_count,
    )
    return cards


# ═══════════════════════════════════════════════════════════════════════════
# Lecture Summariser
# ═══════════════════════════════════════════════════════════════════════════

SUMMARY_PROMPT = """\
You are an expert study assistant helping a student prepare for their exam.

Summarise the following course content clearly and concisely.

Structure your summary EXACTLY as follows (use Markdown headings):

## Key Concepts
- Bullet list of the most important concepts covered

## Main Arguments / Explanations
- Bullet list of the main arguments, processes, or explanations

## Important Definitions
- Term: definition (one per bullet)

## Exam Tips
- What to remember, common pitfalls, likely exam topics

Use clear, plain language a student would understand. Be comprehensive but not padded.

Content ({scope}):
{context}
"""


def generate_summary(
    module_id: str | None   = None,
    week_id: str | None     = None,
    document_id: str | None = None,
    student_id: str         = "",
    scope: str              = "document",
    topic: str | None       = None,
    complexity: str         = "normal",
) -> tuple[str, int]:
    """
    SUM-01/02: Generate a structured Markdown summary.
    Returns (markdown_content, source_doc_count).
    """
    scope_labels = {
        "document": "single document",
        "week":     "weekly lecture materials",
        "module":   "full module",
    }
    query = topic if topic else "key concepts definitions main arguments important facts"
    context, chunk_count = _retrieve_context(
        query=query,
        module_id=module_id,
        week_id=week_id,
        document_id=document_id,
        student_id=student_id,
        top_k=40,   # high top_k — we want comprehensive coverage for summaries
    )

    if not context:
        raise ValueError("No content found to summarise. Upload materials first.")

    prompt  = _with_topic_focus(SUMMARY_PROMPT.format(
        scope=scope_labels.get(scope, scope),
        context=context[:14000],
    ), topic)
    prompt  = _with_complexity(prompt, "summary", complexity)

    llm     = get_llm()
    result  = llm.complete(prompt)
    summary = str(result).strip()

    log.info(
        "summary_generated",
        scope=scope,
        chunks_used=chunk_count,
        summary_length=len(summary),
    )
    return summary, chunk_count
