"""
Usage policies — opt-in quotas and cost budgets for platform "pass" sessions.

Scope (deliberately narrow):
  * A policy can only be attached when an allow-listed platform's API key (server-side) mints a
    session token with an explicit `usage_policy` (today: Learnify's free-course AI Tutor Pass).
  * Sessions WITHOUT a policy — every other platform, Learnify paid courses, API-key calls, and the
    native apps — are never touched by anything in this module.
  * A token minted under a policy is bound to it. If the policy is invalid, expired or revoked the
    token is rejected (fail closed) — it never falls back to unrestricted access.

Enforcement, per guarded AI request (chat / quiz / flashcards / summary):
  1. reserve  — one DB transaction: lock the policy row and reserve a worst-case cost against the
                budget (spent + reserved + estimate <= budget), then atomically bump today's
                counter for the feature only if it is below the limit, and record a reserved event.
  2. call     — the model runs unchanged; token usage is captured from OpenAI's response.
  3. settle   — success: replace the reservation with the measured cost.
                failure: release the daily count; any cost actually incurred is still charged.

Estimation limits (documented): reservations use hard upper bounds — the LLM output cap
(max_tokens=2048) plus a maximum prompt size assuming one token per character — priced for the
configured chat model (unknown models use the most expensive known prices). Measured costs therefore
cannot exceed reservations unless OpenAI's prices or those bounds change. Retrieval-embedding cost is
not reported per request, so a fixed allowance is added. Reservations left by a crashed worker are
charged at their estimate after RESERVATION_STALE_AFTER.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, AsyncIterator, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import llm_usage
from app.config import get_settings
from app.db.models import ApiKey, SessionToken, SessionTokenPolicy, UsageCounter, UsageEvent, UsagePolicy
from app.logging_config import get_logger

log = get_logger(__name__)

PASS_TYPE = "learnify_free_course_pass"
FEATURES  = ("chat", "quiz", "flashcards", "summary")

MAX_TOPIC_CHARS          = 500                    # pass sessions only — keeps prompts bounded
RESERVATION_STALE_AFTER  = timedelta(minutes=10)
EXPIRY_MATCH_TOLERANCE   = timedelta(seconds=1)
EXPIRY_CLOCK_SKEW        = timedelta(minutes=5)

# Must match the LLM's max_tokens (app/agents/llm_factory.get_llm) — guarded by a test.
OUTPUT_MAX_TOKENS = 2048
# Worst-case prompt sizes, assuming 1 token per character (real text is ~4 chars per token):
#   chat: last 10 history messages (5 x 4000-char questions + 5 x 2048-token answers),
#         6 retrieved chunks of 512 tokens, the 4000-char question and the system prompt.
#   quiz/flashcards: context truncated to 12,000 chars + prompt template + 500-char topic.
#   summary: context truncated to 14,000 chars + prompt template + 500-char topic.
INPUT_MAX_TOKENS = {"chat": 40_000, "quiz": 16_000, "flashcards": 16_000, "summary": 18_000}
RETRIEVAL_ALLOWANCE_MICRO_USD = 1_000            # query embeddings (text-embedding-3-large), per request

# USD per 1M tokens == micro-USD per token: (input, output)
MODEL_PRICES = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1":      (2.00, 8.00),
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4o":       (2.50, 10.00),
}
_FALLBACK_PRICES = (max(p[0] for p in MODEL_PRICES.values()), max(p[1] for p in MODEL_PRICES.values()))


# ── Errors ─────────────────────────────────────────────────────────────────

class UsagePolicyError(Exception):
    """Rendered as {"detail": <message>, "code": <code>, ...}; `detail` stays a string for SDK clients."""

    def __init__(self, status_code: int, code: str, message: str,
                 extra: Optional[dict] = None, headers: Optional[dict] = None):
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message
        self.extra, self.headers = extra or {}, headers or {}


# ── Clock / calendar ───────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def local_date(now: datetime, tz: str) -> date:
    return now.astimezone(ZoneInfo(tz)).date()


def next_reset(now: datetime, tz: str) -> datetime:
    """Next local midnight, in UTC (ZoneInfo handles the daylight-saving changes)."""
    zone = ZoneInfo(tz)
    tomorrow = now.astimezone(zone).date() + timedelta(days=1)
    return datetime.combine(tomorrow, time(0, 0), tzinfo=zone).astimezone(timezone.utc)


# ── Pricing ────────────────────────────────────────────────────────────────

def model_prices(model: Optional[str]) -> tuple[float, float]:
    if model:
        if model in MODEL_PRICES:
            return MODEL_PRICES[model]
        matches = [name for name in MODEL_PRICES if model.startswith(f"{name}-")]   # dated snapshots
        if matches:
            return MODEL_PRICES[max(matches, key=len)]
    return _FALLBACK_PRICES


def reservation_micro_usd(feature: str, model: Optional[str]) -> int:
    price_in, price_out = model_prices(model)
    return int(INPUT_MAX_TOKENS[feature] * price_in + OUTPUT_MAX_TOKENS * price_out) + RETRIEVAL_ALLOWANCE_MICRO_USD


def measured_micro_usd(usages: list[dict]) -> Optional[int]:
    """Cost of the captured LLM calls, or None if any call's usage is unknown."""
    if not usages or any(u.get("missing") for u in usages):
        return None
    total = 0.0
    for u in usages:
        price_in, price_out = model_prices(u.get("model") or get_settings().openai_chat_model)
        total += u["input_tokens"] * price_in + u["output_tokens"] * price_out
    return int(round(total))


# ── Policy creation / loading ──────────────────────────────────────────────

def pass_limits() -> dict[str, int]:
    s = get_settings()
    return {"chat": s.pass_daily_chat, "quiz": s.pass_daily_quiz,
            "flashcards": s.pass_daily_flashcards, "summary": s.pass_daily_summary}


def _require_platform(api_key: ApiKey) -> None:
    if api_key.platform not in get_settings().usage_policy_platform_set:
        raise UsagePolicyError(403, "policy_not_permitted", "This integration cannot use usage policies.")


async def attach_policy(
    db: AsyncSession, api_key: ApiKey, *, policy_type: str, external_ref: str,
    expires_at: datetime, user_id: str, user_role: str,
) -> UsagePolicy:
    """
    Create the pass policy on first use, or return the identical existing one. Policies are
    immutable: a different user, type or expiry for the same pass is rejected, and a revoked or
    expired policy can never be re-activated. The caller commits.
    """
    s = get_settings()
    _require_platform(api_key)
    if policy_type != PASS_TYPE:
        raise UsagePolicyError(422, "policy_invalid", "Unsupported usage policy type.")
    if user_role != "student":
        raise UsagePolicyError(422, "policy_invalid", "Usage policies apply to student sessions only.")
    if expires_at.tzinfo is None:
        raise UsagePolicyError(422, "policy_invalid", "expires_at must include a timezone.")

    now = _now()
    if expires_at <= now:
        raise UsagePolicyError(403, "policy_expired", "This AI Tutor Pass has expired.")
    if expires_at > now + timedelta(days=s.pass_max_days) + EXPIRY_CLOCK_SKEW:
        raise UsagePolicyError(422, "policy_invalid", f"A pass cannot last longer than {s.pass_max_days} days.")

    # Insert-if-absent, then lock: concurrent first uses of the same pass converge on one row
    await db.execute(
        pg_insert(UsagePolicy)
        .values(
            id=uuid.uuid4(), api_key_id=api_key.id, platform=api_key.platform, policy_type=policy_type,
            external_ref=external_ref, platform_user_id=user_id, daily_limits=pass_limits(),
            timezone=s.usage_policy_timezone, budget_micro_usd=int(round(s.pass_budget_usd * 1_000_000)),
            spent_micro_usd=0, reserved_micro_usd=0, expires_at=expires_at,
        )
        .on_conflict_do_nothing(constraint="uq_usage_policy_ref")
    )
    policy = (await db.execute(
        select(UsagePolicy)
        .where(UsagePolicy.api_key_id == api_key.id, UsagePolicy.external_ref == external_ref)
        .with_for_update()
    )).scalar_one()

    if (policy.platform_user_id != user_id or policy.policy_type != policy_type
            or abs(policy.expires_at - expires_at) > EXPIRY_MATCH_TOLERANCE):
        raise UsagePolicyError(409, "policy_conflict", "This pass is already registered with different details.")
    _check_active(policy, now)
    return policy


def _check_active(policy: UsagePolicy, now: datetime) -> None:
    if policy.revoked_at is not None:
        raise UsagePolicyError(403, "policy_revoked", "This AI Tutor Pass is no longer active.")
    if policy.expires_at <= now:
        raise UsagePolicyError(403, "policy_expired", "This AI Tutor Pass has expired.")


async def load_policy_for_token(db: AsyncSession, token: SessionToken) -> Optional[UsagePolicy]:
    """
    The policy a session token was minted under, or None for ordinary (unrestricted) tokens.
    Any inconsistency fails closed — the token is rejected rather than treated as unrestricted.
    """
    link = await db.get(SessionTokenPolicy, token.id)
    if link is None:
        return None
    policy = await db.get(UsagePolicy, link.policy_id)
    if (policy is None or policy.api_key_id != token.api_key_id or policy.platform_user_id != token.user_id
            or policy.platform not in get_settings().usage_policy_platform_set):
        raise UsagePolicyError(403, "policy_invalid", "This session's AI Tutor Pass is not valid.")
    _check_active(policy, _now())
    return policy


async def find_policy(db: AsyncSession, api_key: ApiKey, external_ref: str, user_id: str) -> UsagePolicy:
    _require_platform(api_key)
    policy = (await db.execute(
        select(UsagePolicy).where(UsagePolicy.api_key_id == api_key.id, UsagePolicy.external_ref == external_ref)
    )).scalar_one_or_none()
    if policy is None or policy.platform_user_id != user_id:     # never reveal other users' passes
        raise UsagePolicyError(404, "policy_not_found", "No AI Tutor Pass found.")
    return policy


async def revoke_policy(db: AsyncSession, api_key: ApiKey, external_ref: str, user_id: str,
                        reason: Optional[str]) -> UsagePolicy:
    """Idempotent. Tokens minted under the policy stop working immediately."""
    policy = await find_policy(db, api_key, external_ref, user_id)
    if policy.revoked_at is None:
        await db.execute(
            update(UsagePolicy)
            .where(UsagePolicy.id == policy.id, UsagePolicy.revoked_at.is_(None))
            .values(revoked_at=_now(), revoke_reason=(reason or "revoked")[:255])
        )
        await db.commit()
        await db.refresh(policy)
        log.info("usage_policy_revoked", policy_id=str(policy.id), reason=policy.revoke_reason)
    return policy


# ── Enforcement ────────────────────────────────────────────────────────────

@dataclass
class Reservation:
    event_id:   uuid.UUID
    policy_id:  uuid.UUID
    feature:    str
    usage_date: date
    micro_usd:  int


async def _charge_stale(db: AsyncSession, policy_id: uuid.UUID) -> None:
    """Reservations left behind by a crashed worker are charged at their (worst-case) estimate."""
    cutoff = datetime.now(timezone.utc) - RESERVATION_STALE_AFTER
    stale = (await db.execute(
        update(UsageEvent)
        .where(UsageEvent.policy_id == policy_id, UsageEvent.status == "reserved", UsageEvent.created_at < cutoff)
        .values(status="committed", cost_micro_usd=UsageEvent.reserved_micro_usd,
                note="stale_reservation", completed_at=datetime.now(timezone.utc))
        .returning(UsageEvent.reserved_micro_usd)
    )).scalars().all()
    if stale:
        total = sum(stale)
        await db.execute(
            update(UsagePolicy).where(UsagePolicy.id == policy_id).values(
                reserved_micro_usd=UsagePolicy.reserved_micro_usd - total,
                spent_micro_usd=UsagePolicy.spent_micro_usd + total,
            )
        )


async def reserve(db: AsyncSession, policy_id: uuid.UUID, feature: str, course_id: Optional[str]) -> Reservation:
    now = _now()
    estimate = reservation_micro_usd(feature, get_settings().openai_chat_model)
    await _charge_stale(db, policy_id)

    row = (await db.execute(
        update(UsagePolicy)
        .where(
            UsagePolicy.id == policy_id,
            UsagePolicy.revoked_at.is_(None),
            UsagePolicy.expires_at > now,
            UsagePolicy.spent_micro_usd + UsagePolicy.reserved_micro_usd + estimate <= UsagePolicy.budget_micro_usd,
        )
        .values(reserved_micro_usd=UsagePolicy.reserved_micro_usd + estimate)
        .returning(UsagePolicy.daily_limits, UsagePolicy.timezone)
    )).first()
    if row is None:
        await db.rollback()
        policy = await db.get(UsagePolicy, policy_id, populate_existing=True)
        if policy is None:
            raise UsagePolicyError(403, "policy_invalid", "This session's AI Tutor Pass is not valid.")
        _check_active(policy, now)
        raise UsagePolicyError(
            429, "budget_exhausted",
            "You've reached this AI Tutor Pass's fair-use limit. Buy a new pass to keep using the AI tutor.",
            extra={"feature": feature},
        )

    limits, tz = row
    limit = int(limits.get(feature, 0))
    if limit <= 0:
        await db.rollback()
        raise UsagePolicyError(403, "feature_not_included", "This feature isn't included in your AI Tutor Pass.")

    day = local_date(now, tz)
    counter = UsageCounter.__table__
    bumped = (await db.execute(
        pg_insert(UsageCounter)
        .values(policy_id=policy_id, usage_date=day, feature=feature, used=1)
        .on_conflict_do_update(
            index_elements=[counter.c.policy_id, counter.c.usage_date, counter.c.feature],
            set_={"used": counter.c.used + 1},
            where=counter.c.used < limit,
        )
        .returning(counter.c.used)
    )).first()
    if bumped is None:
        await db.rollback()
        resets = next_reset(now, tz)
        noun = {"chat": "tutor questions", "quiz": "quizzes", "flashcards": "flashcard sets",
                "summary": "summaries"}[feature]
        raise UsagePolicyError(
            429, "daily_limit_reached",
            f"You've used today's {limit} {noun}. Your allowance resets at midnight (UK time).",
            extra={"feature": feature, "limit": limit, "resets_at": resets.isoformat()},
            headers={"Retry-After": str(max(1, int((resets - now).total_seconds())))},
        )

    event = UsageEvent(id=uuid.uuid4(), policy_id=policy_id, feature=feature, course_id=course_id,
                       usage_date=day, status="reserved", reserved_micro_usd=estimate)
    db.add(event)
    await db.commit()
    return Reservation(event.id, policy_id, feature, day, estimate)


def _token_totals(usages: list[dict]) -> tuple[Optional[int], Optional[int], Optional[str]]:
    known = [u for u in usages if not u.get("missing")]
    if not known:
        return None, None, None
    return (sum(u["input_tokens"] for u in known), sum(u["output_tokens"] for u in known),
            known[-1].get("model"))


async def settle(db: AsyncSession, res: Reservation, usages: list[dict]) -> int:
    """Success: swap the reservation for the measured cost (the estimate if usage is unknown)."""
    measured = measured_micro_usd(usages)
    cost = res.micro_usd if measured is None else measured + RETRIEVAL_ALLOWANCE_MICRO_USD
    input_tokens, output_tokens, model = _token_totals(usages)
    done = (await db.execute(
        update(UsageEvent)
        .where(UsageEvent.id == res.event_id, UsageEvent.status == "reserved")
        .values(status="committed", cost_micro_usd=cost, input_tokens=input_tokens, output_tokens=output_tokens,
                model=model, note=None if measured is not None else "usage_unavailable",
                completed_at=datetime.now(timezone.utc))
        .returning(UsageEvent.id)
    )).first()
    if done:   # not already charged by the stale sweep
        await db.execute(
            update(UsagePolicy).where(UsagePolicy.id == res.policy_id).values(
                reserved_micro_usd=UsagePolicy.reserved_micro_usd - res.micro_usd,
                spent_micro_usd=UsagePolicy.spent_micro_usd + cost,
            )
        )
    await db.commit()
    if measured is not None and cost > res.micro_usd:
        log.warning("usage_cost_exceeded_reservation", event_id=str(res.event_id), cost=cost, reserved=res.micro_usd)
    return cost


async def release(db: AsyncSession, res: Reservation, usages: list[dict], note: str) -> None:
    """
    Failure: give the request back to the student's daily allowance. Tokens the model actually
    consumed before the failure are still charged to the budget (never under-count real spend).
    """
    measured = measured_micro_usd(usages) if usages else 0
    cost = res.micro_usd if measured is None else measured
    input_tokens, output_tokens, model = _token_totals(usages)
    await db.rollback()
    done = (await db.execute(
        update(UsageEvent)
        .where(UsageEvent.id == res.event_id, UsageEvent.status == "reserved")
        .values(status="released", cost_micro_usd=cost, input_tokens=input_tokens, output_tokens=output_tokens,
                model=model, note=note[:255], completed_at=datetime.now(timezone.utc))
        .returning(UsageEvent.id)
    )).first()
    if done:
        await db.execute(
            update(UsagePolicy).where(UsagePolicy.id == res.policy_id).values(
                reserved_micro_usd=UsagePolicy.reserved_micro_usd - res.micro_usd,
                spent_micro_usd=UsagePolicy.spent_micro_usd + cost,
            )
        )
        await db.execute(
            update(UsageCounter)
            .where(UsageCounter.policy_id == res.policy_id, UsageCounter.usage_date == res.usage_date,
                   UsageCounter.feature == res.feature, UsageCounter.used > 0)
            .values(used=UsageCounter.used - 1)
        )
    await db.commit()


@asynccontextmanager
async def guard(db: AsyncSession, identity: Any, feature: str, course_id: Optional[str],
                topic: Optional[str] = None) -> AsyncIterator[Optional[Reservation]]:
    """
    Enforce the identity's usage policy around one AI request. Identities without a policy
    (everything except pass sessions) pass straight through with no database work.
    """
    policy = getattr(identity, "policy", None)
    if policy is None:
        yield None
        return
    if topic and len(topic) > MAX_TOPIC_CHARS:
        raise UsagePolicyError(422, "topic_too_long", f"Please keep the topic under {MAX_TOPIC_CHARS} characters.")

    res = await reserve(db, policy.id, feature, course_id)
    with llm_usage.capture() as usages:
        try:
            yield res
        except BaseException as exc:
            try:
                await release(db, res, usages, note=f"failed:{type(exc).__name__}")
            except Exception as release_error:  # never mask the original failure
                log.error("usage_release_failed", event_id=str(res.event_id), error=str(release_error))
            raise
    await settle(db, res, usages)


# ── Status ─────────────────────────────────────────────────────────────────

async def policy_status(db: AsyncSession, policy: UsagePolicy, *, include_budget: bool) -> dict:
    now = _now()
    await db.refresh(policy)
    day = local_date(now, policy.timezone)
    used = dict((await db.execute(
        select(UsageCounter.feature, UsageCounter.used)
        .where(UsageCounter.policy_id == policy.id, UsageCounter.usage_date == day)
    )).all())
    remaining_budget = policy.budget_micro_usd - policy.spent_micro_usd - policy.reserved_micro_usd
    model = get_settings().openai_chat_model

    features = {}
    for f in FEATURES:
        limit = int(policy.daily_limits.get(f, 0))
        features[f] = {"limit": limit, "used": used.get(f, 0), "remaining": max(limit - used.get(f, 0), 0),
                       "available": remaining_budget >= reservation_micro_usd(f, model)}

    status = "revoked" if policy.revoked_at else "expired" if policy.expires_at <= now else "active"
    payload = {
        "pass_id":          policy.external_ref,
        "type":             policy.policy_type,
        "status":           status,
        "expires_at":       policy.expires_at.isoformat(),
        "timezone":         policy.timezone,
        "usage_date":       day.isoformat(),
        "resets_at":        next_reset(now, policy.timezone).isoformat(),
        "features":         features,
        "budget_exhausted": remaining_budget < min(reservation_micro_usd(f, model) for f in FEATURES),
    }
    if include_budget:   # server-to-server only; browsers just see budget_exhausted
        payload["budget"] = {
            "total_usd":     policy.budget_micro_usd / 1_000_000,
            "spent_usd":     policy.spent_micro_usd / 1_000_000,
            "reserved_usd":  policy.reserved_micro_usd / 1_000_000,
            "remaining_usd": max(remaining_budget, 0) / 1_000_000,
        }
    return payload
