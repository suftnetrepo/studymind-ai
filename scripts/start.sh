#!/bin/sh
# Use PORT from environment (Render) or default to 8000 (local)
PORT="${PORT:-8000}"

# Run SQL migrations before starting the server.
# psql is not available in the slim image, so we use a small Python one-liner.
# DATABASE_URL must be set in the environment.
echo "==> Running schema migrations..."
python - <<'PYEOF'
import os, sys
import psycopg2

url = os.environ.get("DATABASE_URL", "")
if not url:
    print("WARNING: DATABASE_URL not set — skipping migration", flush=True)
    sys.exit(0)

sql_path = "/app/sql/assignments_tables.sql"
try:
    with open(sql_path) as f:
        sql = f.read()
    conn = psycopg2.connect(url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.close()
    print("Migration OK", flush=True)
except Exception as e:
    print(f"Migration error (non-fatal): {e}", flush=True)
PYEOF

echo "==> Starting uvicorn..."
exec uvicorn app.api.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 2 \
    --log-level info
