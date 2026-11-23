-- Backfill a short join code (CIQ-XXXX) for class modules that have none.
-- Idempotent: only touches class/institution modules whose course_code is NULL or blank, so it is
-- safe to run more than once. Uses the same unambiguous alphabet as the application
-- (no 0/O, 1/I/L) and never reuses a code that already exists on any module.
--
--   psql "$DATABASE_URL" -f sql/backfill_course_codes.sql

DO $$
DECLARE
    alphabet CONSTANT text := 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    m        RECORD;
    candidate text;
    attempts  int;
BEGIN
    FOR m IN
        SELECT id FROM modules
        WHERE access_type <> 'personal' AND (course_code IS NULL OR btrim(course_code) = '')
        ORDER BY created_at
    LOOP
        attempts := 0;
        LOOP
            candidate := 'CIQ-' || (
                SELECT string_agg(substr(alphabet, 1 + floor(random() * length(alphabet))::int, 1), '')
                FROM generate_series(1, 4)
            );
            EXIT WHEN NOT EXISTS (SELECT 1 FROM modules WHERE course_code = candidate);
            attempts := attempts + 1;
            IF attempts >= 50 THEN
                RAISE EXCEPTION 'Could not find a free join code for module %', m.id;
            END IF;
        END LOOP;
        UPDATE modules SET course_code = candidate WHERE id = m.id;
    END LOOP;
END $$;

-- Course codes must be unique so a code (or QR) can never resolve to two modules. Partial index: NULLs
-- (personal modules) are ignored. CONCURRENTLY avoids locking `modules` on a live database; it cannot run
-- inside a transaction, so run this file with plain `psql -f` (not `--single-transaction`).
-- IF NOT EXISTS makes it idempotent. It FAILS if two modules already share a code (e.g. the same CSC109
-- typed by two lecturers) and Postgres then leaves an INVALID index behind that IF NOT EXISTS will skip:
-- check first with
--   SELECT course_code, count(*) FROM modules WHERE course_code IS NOT NULL GROUP BY 1 HAVING count(*) > 1;
-- and, after a failure, `DROP INDEX CONCURRENTLY idx_modules_course_code_unique;` before re-running.
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_modules_course_code_unique
    ON modules (course_code)
    WHERE course_code IS NOT NULL;
