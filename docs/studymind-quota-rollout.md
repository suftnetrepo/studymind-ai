# StudyMind AI quota rollout — release gate (not production ready)

## Approved product rules

- Free: 10 chat, 2 quiz, 2 flashcard, 1 summary generation per UTC calendar month.
- Plus: £3.99 for 28 days, 200 chat, 30 quiz, 30 flashcard, 15 summary generations.
- Web-only purchases; Mobile never initiates payment. Both clients use the same canonical AI user UUID.
- No unlimited AI plan. Module/storage limits (free: 2 modules/25MB; Plus: 20 modules/500MB) are **not yet enforced**.

## Current implementation

- Additive tables: `sql/studymind_entitlements_additive.sql` (not executed in production).
- `GET /api/me/entitlements` shows plan and counters; `enforcement_enabled` is false until rollout.
- `STUDYMIND_QUOTA_ENABLED` defaults to false. When true, scoped chat, general chat, quiz, flashcards and summaries reserve a quota unit in a separate short transaction.
- Streaming chat responds 503 when quotas are enabled, to prevent unmetered bypass until disconnect-safe accounting is implemented.
- Failed AI generation attempts trigger a compensating decrement; this is **not crash-safe or idempotent**. Provider calls may have already incurred cost.
- No Stripe checkout/webhook is connected. Paid entitlement rows must never be inserted by browser/mobile requests.

## Blockers before enablement

1. Run full pytest and staging integration tests, including PostgreSQL concurrent reservations, failure refunds, API error paths and mobile compatibility.
2. Introduce a durable per-request reservation ledger with idempotency keys and crash/disconnect recovery. Current counters alone cannot guarantee exactly-once accounting.
3. Enforce token, retrieval, document storage, and module ceilings; ensure cost budgets are independent of count limits.
4. Implement Stripe server-side Checkout session creation, signed webhook verification, amount/currency/product verification, idempotent event handling, refunds/disputes and entitlement expiry. Never trust user IDs or prices supplied by a browser webhook payload.
5. Verify the staging PDF/DOCX indexing-success and retrieval path with provider credentials, using non-sensitive test documents.
6. Complete verified email-token password recovery; legacy email+new_password endpoint currently returns 410 and is intentionally unavailable.
7. Review and redact historical provider error messages in the database and log retention systems.
8. Perform staged rollout with existing Mobile accounts, monitor errors and cost, and obtain explicit production deployment approval.

Do not set `STUDYMIND_QUOTA_ENABLED=true` in production until these gates pass.
