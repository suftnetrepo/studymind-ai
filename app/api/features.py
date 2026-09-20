"""
AI Features API — Sprint 5.
QUIZ-01–06: Quiz generation, submission, scoring, history
FLASH-01–06: Flashcard deck generation, flip, progress tracking
SUM-01–05: Lecture/module/document summarisation
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.activity.tracker import log_activity
from app.agents.ai_features import (
    generate_flashcards, generate_quiz, generate_summary, score_quiz,
)
from app.auth.dependencies import require_auth
from app.db.engine import get_db
from app.db.models import (
    Document, Flashcard, FlashcardDeck, Module,
    QuizAttempt, QuizQuestion, Summary, Week,
)
from app.db.schemas import (
    FlashcardDeckSchema, FlashcardGenerateRequest,
    FlashcardSchema, FlashcardUpdateRequest,
    QuizAttemptSchema, QuizGenerateRequest,
    QuizSubmitRequest, QuizSubmitResponse,
    SummariseRequest, SummarySchema,
)
from app.db.models import User
from app.logging_config import get_logger

log    = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["features"])


# ═══════════════════════════════════════════════════════════════════════════
# QUIZ
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/quiz/generate", response_model=QuizAttemptSchema, status_code=201)
async def generate_quiz_endpoint(
    req: QuizGenerateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    QUIZ-01/02/03: Generate a quiz from module, week, or document content.
    GPT-4o creates questions with options, correct answers, and explanations.
    """
    # Validate scope
    if not any([req.module_id, req.week_id, req.document_id]):
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: module_id, week_id, document_id",
        )

    title = req.title or _make_title("Quiz", req.module_id, req.week_id, req.document_id, db)

    try:
        questions_data = generate_quiz(
            module_id=str(req.module_id)   if req.module_id   else None,
            week_id=str(req.week_id)       if req.week_id     else None,
            document_id=str(req.document_id) if req.document_id else None,
            student_id=str(current_user.id),
            question_count=req.question_count,
            question_type=req.question_type,
            title=title or "Quiz",
            topic=req.topic,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Persist attempt + questions
    attempt = QuizAttempt(
        id=uuid.uuid4(),
        user_id=current_user.id,
        module_id=req.module_id,
        week_id=req.week_id,
        document_id=req.document_id,
        title=title or f"{req.question_type.upper()} Quiz",
        question_type=req.question_type,
        question_count=len(questions_data),
        status="generated",
    )
    db.add(attempt)
    await db.flush()

    for q in questions_data:
        db.add(QuizQuestion(
            id=uuid.uuid4(),
            attempt_id=attempt.id,
            position=q["position"],
            question=q["question"],
            options=q["options"],
            correct_answer=q["correct_answer"],
            explanation=q["explanation"],
            source_chunk=q["source_chunk"],
        ))

    await db.commit()
    await db.refresh(attempt)
    await log_activity(current_user.id, "quiz", req.module_id)

    # Load with questions
    result = await db.execute(
        select(QuizAttempt)
        .options(selectinload(QuizAttempt.questions))
        .where(QuizAttempt.id == attempt.id)
    )
    return result.scalar_one()


@router.get("/quiz/{attempt_id}", response_model=QuizAttemptSchema)
async def get_quiz(
    attempt_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """QUIZ-04: Retrieve a saved quiz attempt."""
    attempt = await _get_quiz_or_404(attempt_id, current_user.id, db)
    result  = await db.execute(
        select(QuizAttempt)
        .options(selectinload(QuizAttempt.questions))
        .where(QuizAttempt.id == attempt_id)
    )
    return result.scalar_one()


@router.post("/quiz/{attempt_id}/submit", response_model=QuizSubmitResponse)
async def submit_quiz(
    attempt_id: uuid.UUID,
    req: QuizSubmitRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    QUIZ-05/06: Submit answers, score the quiz, return results with explanations.
    Students see which questions were wrong and why, with source citations.
    """
    attempt = await _get_quiz_or_404(attempt_id, current_user.id, db)

    result    = await db.execute(
        select(QuizQuestion)
        .where(QuizQuestion.attempt_id == attempt_id)
        .order_by(QuizQuestion.position)
    )
    questions = result.scalars().all()

    answers_map = {str(a.question_id): a.answer for a in req.answers}
    score, updated_questions = score_quiz(list(questions), answers_map)

    attempt.score        = score
    attempt.status       = "submitted"
    attempt.submitted_at = datetime.now(timezone.utc)

    await db.commit()

    # Reload for response
    result = await db.execute(
        select(QuizAttempt)
        .options(selectinload(QuizAttempt.questions))
        .where(QuizAttempt.id == attempt_id)
    )
    refreshed = result.scalar_one()
    correct   = sum(1 for q in refreshed.questions if q.is_correct)

    return QuizSubmitResponse(
        attempt_id=attempt_id,
        score=score,
        total=len(refreshed.questions),
        correct=correct,
        questions=refreshed.questions,
    )


@router.get("/quiz/history/list", response_model=list[QuizAttemptSchema])
async def quiz_history(
    module_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """QUIZ-07: List past quiz attempts for the current user."""
    query = (
        select(QuizAttempt)
        .where(QuizAttempt.user_id == current_user.id)
        .order_by(QuizAttempt.created_at.desc())
        .limit(50)
    )
    if module_id:
        query = query.where(QuizAttempt.module_id == module_id)
    result = await db.execute(query)
    return result.scalars().all()


# ═══════════════════════════════════════════════════════════════════════════
# FLASHCARDS
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/flashcards/generate", response_model=FlashcardDeckSchema, status_code=201)
async def generate_flashcards_endpoint(
    req: FlashcardGenerateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    FLASH-01/02/03: Generate a flashcard deck from module/week/document content.
    GPT-4o extracts key term-definition pairs grounded in the material.
    """
    if not any([req.module_id, req.week_id, req.document_id]):
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: module_id, week_id, document_id",
        )

    try:
        cards_data = generate_flashcards(
            module_id=str(req.module_id)     if req.module_id   else None,
            week_id=str(req.week_id)         if req.week_id     else None,
            document_id=str(req.document_id) if req.document_id else None,
            student_id=str(current_user.id),
            max_cards=req.max_cards,
            topic=req.topic,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    title = req.title or f"Flashcards — {len(cards_data)} cards"

    deck = FlashcardDeck(
        id=uuid.uuid4(),
        user_id=current_user.id,
        module_id=req.module_id,
        week_id=req.week_id,
        document_id=req.document_id,
        title=title,
        card_count=len(cards_data),
        mastered_count=0,
    )
    db.add(deck)
    await db.flush()

    for c in cards_data:
        db.add(Flashcard(
            id=uuid.uuid4(),
            deck_id=deck.id,
            position=c["position"],
            front=c["front"],
            back=c["back"],
            source_chunk=c["source_chunk"],
            status="new",
        ))

    await db.commit()
    await db.refresh(deck)
    await log_activity(current_user.id, "flashcard", req.module_id)

    result = await db.execute(
        select(FlashcardDeck)
        .options(selectinload(FlashcardDeck.cards))
        .where(FlashcardDeck.id == deck.id)
    )
    return result.scalar_one()


@router.get("/flashcards", response_model=list[FlashcardDeckSchema])
async def list_flashcard_decks(
    module_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """List all flashcard decks for the current user."""
    query = (
        select(FlashcardDeck)
        .options(selectinload(FlashcardDeck.cards))
        .where(FlashcardDeck.user_id == current_user.id)
        .order_by(FlashcardDeck.created_at.desc())
    )
    if module_id:
        query = query.where(FlashcardDeck.module_id == module_id)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/flashcards/{deck_id}", response_model=FlashcardDeckSchema)
async def get_flashcard_deck(
    deck_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """FLASH-03: Get a full deck with all cards."""
    result = await db.execute(
        select(FlashcardDeck)
        .options(selectinload(FlashcardDeck.cards))
        .where(
            FlashcardDeck.id      == deck_id,
            FlashcardDeck.user_id == current_user.id,
        )
    )
    deck = result.scalar_one_or_none()
    if not deck:
        raise HTTPException(status_code=404, detail="Flashcard deck not found")
    return deck


@router.patch("/flashcards/{deck_id}/cards/{card_id}", response_model=FlashcardSchema)
async def update_flashcard_status(
    deck_id: uuid.UUID,
    card_id: uuid.UUID,
    req: FlashcardUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    FLASH-04/05: Update a card's status (new → learning → mastered).
    Also updates deck mastered_count.
    """
    # Verify deck ownership
    deck_result = await db.execute(
        select(FlashcardDeck).where(
            FlashcardDeck.id      == deck_id,
            FlashcardDeck.user_id == current_user.id,
        )
    )
    deck = deck_result.scalar_one_or_none()
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found")

    card_result = await db.execute(
        select(Flashcard).where(
            Flashcard.id      == card_id,
            Flashcard.deck_id == deck_id,
        )
    )
    card = card_result.scalar_one_or_none()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    old_status  = card.status
    card.status = req.status

    # Recalculate mastered count
    if old_status != "mastered" and req.status == "mastered":
        deck.mastered_count = min(deck.mastered_count + 1, deck.card_count)
    elif old_status == "mastered" and req.status != "mastered":
        deck.mastered_count = max(deck.mastered_count - 1, 0)

    deck.last_studied_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(card)
    return card


@router.delete("/flashcards/{deck_id}", status_code=204)
async def delete_flashcard_deck(
    deck_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    result = await db.execute(
        select(FlashcardDeck).where(
            FlashcardDeck.id      == deck_id,
            FlashcardDeck.user_id == current_user.id,
        )
    )
    deck = result.scalar_one_or_none()
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found")
    await db.delete(deck)
    await db.commit()


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARISER
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/summarise", response_model=SummarySchema, status_code=201)
async def summarise_endpoint(
    req: SummariseRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    SUM-01/02/03: Generate a structured Markdown summary.
    Scope: document | week | module.
    SUM-05: Summary cites source documents.
    """
    if not any([req.module_id, req.week_id, req.document_id]):
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: module_id, week_id, document_id",
        )

    try:
        content, chunk_count = generate_summary(
            module_id=str(req.module_id)     if req.module_id   else None,
            week_id=str(req.week_id)         if req.week_id     else None,
            document_id=str(req.document_id) if req.document_id else None,
            student_id=str(current_user.id),
            scope=req.scope,
            topic=req.topic,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    summary = Summary(
        id=uuid.uuid4(),
        user_id=current_user.id,
        module_id=req.module_id,
        week_id=req.week_id,
        document_id=req.document_id,
        scope=req.scope,
        content=content,
        source_doc_count=chunk_count,
    )
    db.add(summary)
    await db.commit()
    await db.refresh(summary)
    await log_activity(current_user.id, "summary", req.module_id)

    log.info(
        "summary_saved",
        summary_id=str(summary.id),
        scope=req.scope,
        length=len(content),
    )
    return summary


@router.get("/summarise", response_model=list[SummarySchema])
async def list_summaries(
    module_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """List all saved summaries for the current user."""
    query = (
        select(Summary)
        .where(Summary.user_id == current_user.id)
        .order_by(Summary.created_at.desc())
        .limit(50)
    )
    if module_id:
        query = query.where(Summary.module_id == module_id)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/summarise/{summary_id}", response_model=SummarySchema)
async def get_summary(
    summary_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    result = await db.execute(
        select(Summary).where(
            Summary.id      == summary_id,
            Summary.user_id == current_user.id,
        )
    )
    summary = result.scalar_one_or_none()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    return summary


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_title(prefix, module_id, week_id, document_id, db) -> str:
    """Best-effort title from scope IDs."""
    if week_id:
        return f"{prefix} — Week {week_id}"
    if module_id:
        return f"{prefix} — Module"
    if document_id:
        return f"{prefix} — Document"
    return prefix


async def _get_quiz_or_404(
    attempt_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> QuizAttempt:
    result  = await db.execute(
        select(QuizAttempt).where(
            QuizAttempt.id      == attempt_id,
            QuizAttempt.user_id == user_id,
        )
    )
    attempt = result.scalar_one_or_none()
    if not attempt:
        raise HTTPException(status_code=404, detail="Quiz attempt not found")
    return attempt
