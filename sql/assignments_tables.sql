-- Assignments feature tables — idempotent (CREATE TABLE IF NOT EXISTS)
-- Run at startup before uvicorn to guarantee tables exist in production.

CREATE TABLE IF NOT EXISTS assignments (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module_id            UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    created_by           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title                VARCHAR(255) NOT NULL,
    description          TEXT,
    due_at               TIMESTAMPTZ NOT NULL,
    brief_cloudinary_id  VARCHAR(512) NOT NULL,
    brief_cloudinary_url TEXT NOT NULL,
    brief_filename       VARCHAR(512) NOT NULL,
    brief_file_size      INTEGER NOT NULL DEFAULT 0,
    scheme_cloudinary_id  VARCHAR(512) NOT NULL,
    scheme_cloudinary_url TEXT NOT NULL,
    scheme_filename       VARCHAR(512) NOT NULL,
    scheme_file_size      INTEGER NOT NULL DEFAULT 0,
    max_score            INTEGER NOT NULL DEFAULT 100,
    status               VARCHAR(32) NOT NULL DEFAULT 'active',
    grading_prompt       TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_assignments_module_id ON assignments(module_id);

CREATE TABLE IF NOT EXISTS assignment_submissions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assignment_id  UUID NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
    student_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    cloudinary_id  VARCHAR(512) NOT NULL,
    cloudinary_url TEXT NOT NULL,
    filename       VARCHAR(512) NOT NULL,
    file_size      INTEGER NOT NULL DEFAULT 0,
    is_late        BOOLEAN NOT NULL DEFAULT FALSE,
    status         VARCHAR(32) NOT NULL DEFAULT 'pending',
    score          FLOAT,
    strengths      JSONB,
    weaknesses     JSONB,
    recommendations JSONB,
    grading_notes  TEXT,
    submitted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    graded_at      TIMESTAMPTZ,
    CONSTRAINT uq_one_submission_per_student UNIQUE (assignment_id, student_id)
);

CREATE INDEX IF NOT EXISTS ix_assignment_submissions_assignment_id
    ON assignment_submissions(assignment_id);

-- ── Idempotent ALTER TABLE to fix column types on pre-existing tables ────────
-- These statements are safe to re-run; they no-op if the column already has
-- the right type. They correct tables that were created manually (or by an
-- earlier version of this file) with wrong types:
--   score        was INTEGER  → must be FLOAT  (ORM: Float)
--   strengths    was TEXT     → must be JSONB  (ORM: JSONB)
--   weaknesses   was TEXT     → must be JSONB
--   recommendations was TEXT  → must be JSONB

DO $$
BEGIN
    -- score: INTEGER → DOUBLE PRECISION (FLOAT)
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'assignment_submissions'
          AND column_name = 'score'
          AND data_type = 'integer'
    ) THEN
        ALTER TABLE assignment_submissions ALTER COLUMN score TYPE DOUBLE PRECISION;
    END IF;

    -- strengths: TEXT → JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'assignment_submissions'
          AND column_name = 'strengths'
          AND data_type = 'text'
    ) THEN
        ALTER TABLE assignment_submissions
            ALTER COLUMN strengths TYPE JSONB USING strengths::jsonb;
    END IF;

    -- weaknesses: TEXT → JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'assignment_submissions'
          AND column_name = 'weaknesses'
          AND data_type = 'text'
    ) THEN
        ALTER TABLE assignment_submissions
            ALTER COLUMN weaknesses TYPE JSONB USING weaknesses::jsonb;
    END IF;

    -- recommendations: TEXT → JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'assignment_submissions'
          AND column_name = 'recommendations'
          AND data_type = 'text'
    ) THEN
        ALTER TABLE assignment_submissions
            ALTER COLUMN recommendations TYPE JSONB USING recommendations::jsonb;
    END IF;
END
$$;
