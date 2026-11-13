# StudyMind Web ↔ AI backend contract (pre-production)

## Canonical ownership

- StudyMind AI PostgreSQL `users.id` (UUID) is the identity used by Mobile and Web.
- Web must never silently join legacy Prisma accounts to AI accounts by email alone.
- Module, enrolment, material and AI usage permissions are checked in AI, not trusted from the browser.
- Existing Mobile users and content remain in place; no destructive migration.

## Implemented AI endpoints

- `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/refresh`, `GET /api/auth/me`.
- `GET /api/auth/learning-overview` is additive on this feature branch, not yet deployed.
- `GET /api/modules`, `POST /api/modules`, `GET /api/modules/{id}` and module documents endpoints already exist in the AI service. Confirm request/response schemas and authorization in integration tests before rollout.

## Usage and purchases: do not conflate the two identity systems

The existing `/v1/usage/policy` implementation is an opt-in, API-key/session-token policy for allow-listed platform integrations (currently Learnify passes). It explicitly does **not** enforce quotas for native Mobile sessions. Reusing it as-is would allow Mobile to bypass paid Web quotas. Do not advertise shared entitlements until canonical JWT-based AI endpoints enforce them.

### Required before production

1. Define a canonical entitlement table keyed by AI `users.id`, with Stripe checkout/payment/refund/dispute references and a unique idempotency key per Stripe event.
2. Create a server-to-server-only activation/revocation API (verified caller, never browser-accessible). Verify Stripe payment status and ownership before granting access. Web cannot grant credits by merely claiming success.
3. Define feature-level free allowance and paid credits, plus the reset period and expiry semantics. Do not assume existing Learnify pass daily quotas apply to StudyMind users.
4. At the AI entry points used by **both Mobile and Web**, reserve usage atomically before LLM calls, settle on success, release on failure, and enforce limits server-side. Guard retries and concurrent calls.
5. Expose authenticated read-only `GET /api/.../me/entitlements` with remaining allowances and expiry; return the same response to Mobile and Web.
6. Regression test legacy Mobile flows, free-to-paid transitions, refunds, disputed payments, expiry, concurrency, account ownership and migration of existing users.
7. Use feature flags and staged rollout; never deploy schema or billing changes to the live service without backup, rehearsal and approval.

## Security release blocker

The live legacy `POST /api/auth/forgot-password` accepts an email and replacement password without verifying email ownership. Security patch is in draft PR #5. A complete fix requires a single-use, expiring, hashed recovery token delivered to a verified mailbox, uniform responses for existing/non-existing emails, rate limiting, and session revocation after successful reset. Do not enable the legacy reset flow on Web.
