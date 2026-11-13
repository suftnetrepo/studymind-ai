-- StudyMind canonical entitlement foundation. REVIEW before running against live DB.
-- Additive only: no existing users, modules or documents are modified.
BEGIN;
CREATE TABLE IF NOT EXISTS studymind_entitlements (
 id uuid PRIMARY KEY,
 user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 stripe_checkout_session_id varchar(255) NOT NULL,
 stripe_payment_intent_id varchar(255),
 status varchar(32) NOT NULL DEFAULT 'active',
 feature_limits jsonb NOT NULL DEFAULT '{}'::jsonb,
 starts_at timestamptz NOT NULL,
 expires_at timestamptz NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 CONSTRAINT uq_studymind_checkout_session UNIQUE (stripe_checkout_session_id),
 CONSTRAINT ck_studymind_entitlement_window CHECK (expires_at > starts_at)
);
CREATE INDEX IF NOT EXISTS ix_studymind_entitlements_user
 ON studymind_entitlements(user_id, expires_at);
CREATE TABLE IF NOT EXISTS studymind_usage_counters (
 user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 feature varchar(32) NOT NULL,
 period_start timestamptz NOT NULL,
 used integer NOT NULL DEFAULT 0 CHECK (used >= 0),
 PRIMARY KEY (user_id, feature, period_start)
);
COMMIT;
