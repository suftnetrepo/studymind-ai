# StudyMind Stripe Checkout — implementation status

## Implemented (feature branch only)
- Authenticated POST /api/billing/checkout; fixed £3.99 GBP one-time payment, no subscription.
- Canonical user ID and email supplied from JWT-authenticated backend identity.
- Test-mode keys only; STUDYMIND_BILLING_ENABLED defaults false.
- Checkout requires STUDYMIND_STRIPE_SECRET_KEY, STUDYMIND_STRIPE_WEBHOOK_SECRET, STUDYMIND_CHECKOUT_SUCCESS_URL and STUDYMIND_CHECKOUT_CANCEL_URL.
- Checkout does **not** grant access or trust browser success redirects.

## Required before enabling
1. Implement signed Stripe webhook verification using the raw request body and tolerance-checked timestamp.
2. Reconcile Checkout Session via Stripe API and require payment_status=paid, currency=gbp, amount_total=399, mode=payment, and exact metadata product/user identity.
3. Atomically create one 28-day entitlement per verified session with duplicate-event and concurrency tests.
4. Define overlapping pass, refund, dispute, partial refund and expired-pass policy; implement and test it.
5. Add Stripe CLI test-mode integration, failed payment, duplicate webhook, out-of-order webhook and replay tests.
6. Audit additive SQL and account deletion against entitlements and usage counters.
7. Validate real staging setup and obtain explicit approval before enabling billing or deployment.

**Do not turn on STUDYMIND_BILLING_ENABLED.** No webhook exists yet; Plus must remain Coming soon on Web.
