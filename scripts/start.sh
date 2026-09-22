#!/bin/sh
# Use PORT from environment (Render) or default to 8000 (local)
PORT="${PORT:-8000}"
exec uvicorn app.api.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 2 \
    --log-level info
