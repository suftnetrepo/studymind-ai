"""Application import/startup check (run in CI and locally).

Fails if the app cannot be imported (syntax/indentation/import errors), or if a route the clients depend on is not
actually registered. Registration is read from the generated OpenAPI schema, not from source text, so a handler
that is defined but unreachable (e.g. accidentally nested inside another function) is caught.

Needs no database, network or real API keys:  OPENAI_API_KEY=x python scripts/check_app_startup.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `app` imports from anywhere

os.environ.setdefault("OPENAI_API_KEY", "ci-placeholder-not-a-real-key")
os.environ.setdefault("JWT_SECRET_KEY", "ci-placeholder-secret-minimum-32-characters")
os.environ.setdefault("TYPESENSE_API_KEY", "ci-placeholder")

REQUIRED = [
    ("post", "/api/chat"), ("post", "/api/chat/stream"), ("post", "/api/chat/general"),
    ("post", "/api/chat/transcribe"), ("post", "/api/chat/extract-image"),
    ("post", "/api/quiz/generate"), ("post", "/api/quiz/{attempt_id}/submit"),
    ("post", "/api/flashcards/generate"), ("post", "/api/summarise"),
    ("get", "/api/modules"), ("post", "/api/modules/join"), ("get", "/api/health"),
]


def main() -> int:
    from app.api.main import app  # noqa: PLC0415 — import is the check

    paths = app.openapi()["paths"]
    missing = [f"{m.upper()} {p}" for m, p in REQUIRED if m not in paths.get(p, {})]
    if missing:
        print("Routes missing from the running application:\n  " + "\n  ".join(missing))
        return 1
    print(f"OK: application imported; {len(paths)} paths; all {len(REQUIRED)} required routes registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
