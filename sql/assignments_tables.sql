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
    score          INTEGER,
    strengths      TEXT,
    weaknesses     TEXT,
    recommendations TEXT,
    grading_notes  TEXT,
    submitted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    graded_at      TIMESTAMPTZ,
    CONSTRAINT uq_one_submission_per_student UNIQUE (assignment_id, student_id)
);

CREATE INDEX IF NOT EXISTS ix_assignment_submissions_assignment_id
    ON assignment_submissions(assignment_id);
