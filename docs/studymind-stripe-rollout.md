# StudyMind Stripe Checkout — implementation status

## Implemented (feature branch only)
- Authenticated POST /api/billing/checkout; fixed £3.99 GBP one-time payment, no subscription.
- Canonical user ID and email supplied from JWT-authenticated backend identity.
- Test-mode keys only; STUDYMIND_BILLING_ENABLED defaults false.
- Checkout requires STUDYMIND_STRIPE_SECRET_KEY, STUDYMIND_STRIPE_WEBHOOK_SECRET, STUDYMIND_CHECKOUT_SUCCESS_URL and STUDYMIND_CHECKOUT_CANCEL_URL.
- Checkout does **not** grant access or trust browser success redirects.

## Additional implementation on feature branch\n- POST /api/billing/stripe/webhook verifies the raw-body Stripe v1 HMAC with a 5-minute timestamp window.\n- Completed Checkout is fetched again from Stripe and checked for test mode, paid status, GBP 399, canonical user identity and product metadata.\n- PostgreSQL unique Checkout Session constraint plus ON CONFLICT DO NOTHING makes repeated activation idempotent.\n- Refund and dispute events are intentionally not processed yet. **Billing must remain disabled.**\n- Pure validation tests added; not yet executed against a real Stripe test webhook or PostgreSQL.\n\n## Required before enabling
1. Run signed Stripe webhook verification against Stripe CLI test-mode events and verify retries.
2. Test real Stripe Checkout Session reconciliation and reject incorrect currency, amount, identity and product.
3. Run PostgreSQL duplicate-event and concurrent-delivery tests proving one 28-day entitlement per verified session.
4. Define overlapping pass, refund, dispute, partial refund and expired-pass policy; implement and test it.
5. Add Stripe CLI test-mode integration, failed payment, duplicate webhook, out-of-order webhook and replay tests.
6. Audit additive SQL and account deletion against entitlements and usage counters.
7. Validate real staging setup and obtain explicit approval before enabling billing or deployment.

**Do not turn on STUDYMIND_BILLING_ENABLED.** No webhook exists yet; Plus must remain Coming soon on Web.
