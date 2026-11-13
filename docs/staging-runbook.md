# Staging runbook (AI backend)

Nothing here has been executed against a real staging host. Deployment needs explicit approval.

## Topology and isolation

`docker-compose.staging.yml` runs postgres, typesense, budget-proxy and api under the project name `studymind-staging`.
It shares **no** container name, port, volume, database, Typesense instance or env file with `docker-compose.prod.yml`
(checked by rendering both; all published ports are bound to 127.0.0.1). Do not point staging at production's database,
Typesense (the collection name `document_chunks` is fixed, so a shared instance would mix data), Cloudinary or Stripe.

The API holds only a dummy OpenAI key. `budget-proxy` holds the real key, enforces the dollar ceiling (`STAGING_MAX_USD`,
default 5) and call caps, serialises requests, and **persists its ledger** (`LEDGER_FILE`), so a restart cannot reset the
ceiling. A dashboard budget alert is not a cap; an OpenAI project hard limit is an additional, separately verified control.

## Before deploying (all must pass)

1. `python scripts/verify_staging_env.py --forbid-host <prod db host> --forbid-host <prod typesense host> --expect-model gpt-4.1-mini --expect-threshold 0.20 --max-llm-retries 1 --max-embed-retries 0 --proxy-url http://127.0.0.1:18300` prints `READY`. It fails if: APP_ENV is not staging; the database name lacks "staging"; a production host is configured; Typesense/JWT secrets are placeholders; OpenAI is not routed through the proxy or the API holds a real key; the model is unpriced or unexpected; retries/threshold differ; a live Stripe key is set; Cloudinary's folder is not staging-specific. It prints no secret values.
2. `python scripts/print_model_config.py` shows model, prices, threshold 0.2, retries 1/0, vector k 100.
3. CI green on the exact commit; the 86-check security suite passes against the candidate.

## Migrations

The schema is created at startup by `create_all_tables()` (advisory-lock protected). Compared with `main`, the candidate adds
**two new tables** (`studymind_entitlements`, `studymind_usage_counters`) and changes **no existing table**, so `create_all`
is sufficient. The reviewed manual equivalent is `sql/studymind_entitlements_additive.sql` (additive, transactional). The
Typesense schema is unchanged: no reindex. Re-check with a schema diff against the exact base before any deployment.

## Deploy (requires approval)

`docker compose -f docker-compose.staging.yml --env-file .env.staging up -d --build`, then `GET :18000/api/health` shows
postgresql and typesense true; `GET :18300/__ledger` shows `maxUsd <= 5`, `concurrency 1`.

## Rollback

* Code: redeploy the previous image/commit (`git checkout <previous sha>`, `up -d --build api`). The two new tables are additive and unused by older code; leave them.
* Behaviour only: unset `SIMILARITY_THRESHOLD` / `RETRIEVAL_VECTOR_K` (defaults 0.25 / 100) and restart `api`. No data change.
* Data: staging data is disposable. `docker compose -f docker-compose.staging.yml down -v` removes only `studymind-staging_*` volumes. Never run it with the production file or project name.
* Cost: stop `budget-proxy` to halt all paid calls immediately.

## Monitoring the ceiling

`curl -s 127.0.0.1:18300/__ledger` gives calls, tokens, estimated spend and refusals (no prompts or keys). Refusals with
`budget_proxy_limit` mean the ceiling or caps held. Reset only deliberately: `curl -X DELETE 127.0.0.1:18300/__ledger`.

## Production-config findings (not changed here)

`docker-compose.prod.yml` hard-codes and commits the Typesense API key (`rag-typesense-key`), publishes Typesense on all
interfaces (`8108:8108`) with `--enable-cors`, and the API allows CORS origin `*`. Rotate/lock these separately; staging does not repeat them.
