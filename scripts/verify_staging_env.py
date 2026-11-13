"""Pre-deployment verification of a STAGING environment. Fails closed. Prints no secret values.

    python scripts/verify_staging_env.py --forbid-host <production db host> --forbid-host <production typesense host> \
        --expect-model gpt-4.1-mini --expect-threshold 0.20 --max-llm-retries 1 --max-embed-retries 0 [--proxy-url http://host:18300]

Run it in the environment that will run the staging API (same env vars / .env file). Exit code 1 if any check FAILs.
It cannot know your production hostnames: pass them with --forbid-host (or STAGING_FORBIDDEN_HOSTS=a,b) so a staging
config that points at production is caught.
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PLACEHOLDERS = re.compile(r"(changeme|change-me|placeholder|example|secret-key|test-key|local-test|local-integration|xyz|^x+$|^0+$)", re.I)
REAL_OPENAI_KEY = re.compile(r"^sk-(proj-|svcacct-|admin-)?[A-Za-z0-9_-]{30,}$")


def _host_of(value: str) -> str:
    value = value or ""
    return (urlsplit(value if "://" in value else f"//{value}").hostname or value).lower()


def run_checks(s, environ, opts) -> list[tuple[str, str, str]]:
    """Returns [(level, check, detail)] with level in PASS / WARN / FAIL. `s` is a Settings instance."""
    from app.usage import policy as usage

    out: list[tuple[str, str, str]] = []
    add = lambda level, name, detail="": out.append((level, name, detail))  # noqa: E731
    forbidden = {h.lower() for h in opts.forbid_host if h}

    # 1. environment name
    add("PASS" if s.app_env == "staging" else "FAIL", "APP_ENV is 'staging'", f"APP_ENV={s.app_env!r}")

    # 2. database isolation
    dsn = urlsplit(s.database_url) if s.database_url else None
    db_name = (dsn.path.lstrip("/") if dsn else s.postgres_db) or ""
    db_host = (dsn.hostname if dsn else s.postgres_host) or ""
    add("PASS" if "staging" in db_name.lower() else "FAIL", "database name contains 'staging'", f"db={db_name!r}")
    add("FAIL" if db_host.lower() in forbidden else "PASS", "database host is not a forbidden (production) host", f"host={db_host!r}")
    if not forbidden:
        add("WARN", "no --forbid-host given", "cannot prove the database/Typesense hosts differ from production")

    # 3. Typesense isolation (the collection name 'document_chunks' is fixed, so a shared instance would mix data)
    ts_host = _host_of(s.typesense_url) if s.typesense_url else (s.typesense_host or "")
    add("PASS" if ts_host else "FAIL", "Typesense host configured", f"host={ts_host!r}")
    add("FAIL" if ts_host.lower() in forbidden else "PASS", "Typesense host is not a forbidden (production) host", f"host={ts_host!r}")
    k = s.typesense_api_key or ""
    add("PASS" if len(k) >= 16 and not PLACEHOLDERS.search(k) else "FAIL", "Typesense API key is set and not a placeholder", f"length={len(k)}")

    # 4. secrets
    j = s.jwt_secret_key or ""
    add("PASS" if len(j) >= 32 and not PLACEHOLDERS.search(j) else "FAIL", "JWT secret is >=32 chars and not a placeholder", f"length={len(j)}")

    # 5. OpenAI: the cost ceiling lives in the budget proxy, so the backend must reach OpenAI only through it
    base = environ.get("OPENAI_API_BASE") or environ.get("OPENAI_BASE_URL") or ""
    via_proxy = bool(base) and _host_of(base) != "api.openai.com"
    add("PASS" if via_proxy else "FAIL", "OpenAI base URL points at the budget proxy (ceiling enforced)", f"base={base or 'unset (direct to api.openai.com)'}")
    holds_real_key = bool(REAL_OPENAI_KEY.match(s.openai_api_key or ""))
    add("FAIL" if holds_real_key else "PASS", "backend does NOT hold a real OpenAI key (proxy holds it)", "a real-looking key is configured" if holds_real_key else "dummy/empty")
    if opts.proxy_url:
        try:
            with urllib.request.urlopen(opts.proxy_url.rstrip("/") + "/__ledger", timeout=5) as r:
                lim = json.loads(r.read())["limits"]
            ok = lim["maxUsd"] <= opts.max_usd and lim["concurrency"] == 1
            add("PASS" if ok else "FAIL", f"proxy ceiling <= ${opts.max_usd} and concurrency 1", json.dumps(lim))
        except Exception as e:  # noqa: BLE001
            add("FAIL", "budget proxy reachable", f"{type(e).__name__}")

    # 6. model, pricing, retrieval, retries
    priced = s.openai_chat_model in usage.MODEL_PRICES or any(s.openai_chat_model.startswith(f"{m}-") for m in usage.MODEL_PRICES)
    add("PASS" if priced else "FAIL", "chat model is in the price table", f"model={s.openai_chat_model!r}")
    if opts.expect_model:
        add("PASS" if s.openai_chat_model == opts.expect_model else "FAIL", f"chat model is {opts.expect_model}", f"model={s.openai_chat_model!r}")
    if opts.expect_threshold is not None:
        add("PASS" if abs(s.similarity_threshold - opts.expect_threshold) < 1e-9 else "FAIL", f"similarity threshold is {opts.expect_threshold}", f"value={s.similarity_threshold}")
    add("PASS" if s.openai_llm_max_retries <= opts.max_llm_retries else "FAIL", f"LLM retries <= {opts.max_llm_retries}", f"value={s.openai_llm_max_retries}")
    add("PASS" if s.openai_embed_max_retries <= opts.max_embed_retries else "FAIL", f"embedding retries <= {opts.max_embed_retries}", f"value={s.openai_embed_max_retries}")
    add("PASS" if s.retrieval_vector_k >= s.top_k_retrieval else "FAIL", "vector candidate depth >= top_k", f"k={s.retrieval_vector_k} top_k={s.top_k_retrieval}")

    # 7. no live payment keys; billing stays off until staging webhook validation
    sk = s.studymind_stripe_secret_key or ""
    live = sk.startswith(("sk_live", "rk_live"))
    add("FAIL" if live else "PASS", "no live Stripe key", "a LIVE key is configured" if live else ("test/empty" if sk else "empty"))
    add("PASS" if not s.studymind_billing_enabled or sk.startswith(("sk_test", "rk_test")) else "FAIL", "billing is off, or uses a test-mode key")

    # 8. document storage must not be production's
    if s.cloudinary_configured:
        add("PASS" if "staging" in (s.cloudinary_folder or "").lower() else "FAIL", "Cloudinary folder is staging-specific", f"folder={s.cloudinary_folder!r}")
    else:
        add("PASS", "Cloudinary not configured (staging cannot write to production storage)")

    # 9. outbound email must not reach real users
    add("PASS" if not s.password_recovery_enabled else "WARN", "password-recovery email is off", "enabled: SMTP must be a sink/test inbox" if s.password_recovery_enabled else "off")
    add("PASS", "informational: quota enforcement", "ON" if s.studymind_quota_enabled else "off")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forbid-host", action="append", default=[x for x in os.environ.get("STAGING_FORBIDDEN_HOSTS", "").split(",") if x])
    ap.add_argument("--expect-model"); ap.add_argument("--expect-threshold", type=float)
    ap.add_argument("--max-llm-retries", type=int, default=1); ap.add_argument("--max-embed-retries", type=int, default=1)
    ap.add_argument("--proxy-url"); ap.add_argument("--max-usd", type=float, default=5.0)
    opts = ap.parse_args(argv)
    os.environ.setdefault("OPENAI_API_KEY", "")
    from app.config import get_settings

    results = run_checks(get_settings(), os.environ, opts)
    width = max(len(n) for _, n, _ in results)
    for level, name, detail in results:
        print(f"{level:5} {name.ljust(width)}  {detail}")
    fails = sum(1 for lv, _, _ in results if lv == "FAIL")
    print(f"\n{'NOT READY' if fails else 'READY'}: {fails} FAIL, {sum(1 for lv, _, _ in results if lv == 'WARN')} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
