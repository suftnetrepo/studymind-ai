"""The staging verifier must fail closed on every way a staging config could touch production or bypass the cost ceiling."""
import argparse
import importlib.util
from pathlib import Path

import pytest

from app.config import Settings

spec = importlib.util.spec_from_file_location("verify_staging_env", Path(__file__).resolve().parent.parent / "scripts" / "verify_staging_env.py")
vse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vse)

GOOD = dict(
    app_env="staging", postgres_db="studymind_staging", postgres_host="db.staging.internal", typesense_host="ts.staging.internal",
    typesense_api_key="k" * 40, jwt_secret_key="Zq" * 24, openai_api_key="dummy-backend-key-not-real",
    openai_chat_model="gpt-4.1-mini", similarity_threshold=0.20, openai_llm_max_retries=1, openai_embed_max_retries=0,
    studymind_stripe_secret_key="", studymind_billing_enabled=False, cloudinary_cloud_name="", password_recovery_enabled=False,
)
GOOD_ENV = {"OPENAI_API_BASE": "http://budget-proxy:18300/v1"}


def opts(**kw):
    base = dict(forbid_host=["prod-db.example.com", "prod-ts.example.com"], expect_model="gpt-4.1-mini", expect_threshold=0.20,
                max_llm_retries=1, max_embed_retries=0, proxy_url=None, max_usd=5.0)
    base.update(kw)
    return argparse.Namespace(**base)


def run(settings_over=None, env=None, **o):
    s = Settings(_env_file=None, **{**GOOD, **(settings_over or {})})
    return vse.run_checks(s, GOOD_ENV if env is None else env, opts(**o))


def fails(results):
    return {name for level, name, _ in results if level == "FAIL"}


def test_a_correct_staging_config_is_ready():
    assert fails(run()) == set()


@pytest.mark.parametrize("over,env,expected", [
    ({"app_env": "production"}, None, "APP_ENV is 'staging'"),
    ({"app_env": "development"}, None, "APP_ENV is 'staging'"),
    ({"postgres_db": "studymind"}, None, "database name contains 'staging'"),
    ({"postgres_host": "prod-db.example.com"}, None, "database host is not a forbidden (production) host"),
    ({"typesense_host": "prod-ts.example.com"}, None, "Typesense host is not a forbidden (production) host"),
    ({"typesense_api_key": "xyz"}, None, "Typesense API key is set and not a placeholder"),
    ({"typesense_api_key": "local-test-key" * 2}, None, "Typesense API key is set and not a placeholder"),
    ({"jwt_secret_key": "short"}, None, "JWT secret is >=32 chars and not a placeholder"),
    ({"jwt_secret_key": "local-integration-secret-32-chars-minimum-xx"}, None, "JWT secret is >=32 chars and not a placeholder"),
    ({}, {}, "OpenAI base URL points at the budget proxy (ceiling enforced)"),
    ({}, {"OPENAI_API_BASE": "https://api.openai.com/v1"}, "OpenAI base URL points at the budget proxy (ceiling enforced)"),
    ({"openai_api_key": "sk-proj-" + "A" * 60}, None, "backend does NOT hold a real OpenAI key (proxy holds it)"),
    ({"openai_chat_model": "gpt-4o"}, None, "chat model is gpt-4.1-mini"),
    ({"openai_chat_model": "gpt-9-future"}, None, "chat model is in the price table"),
    ({"similarity_threshold": 0.25}, None, "similarity threshold is 0.2"),
    ({"openai_llm_max_retries": 3}, None, "LLM retries <= 1"),
    ({"openai_embed_max_retries": 10}, None, "embedding retries <= 0"),
    ({"studymind_stripe_secret_key": "sk_live_" + "a" * 30}, None, "no live Stripe key"),
    ({"studymind_billing_enabled": True, "studymind_stripe_secret_key": ""}, None, "billing is off, or uses a test-mode key"),
    ({"cloudinary_cloud_name": "c", "cloudinary_api_key": "k", "cloudinary_api_secret": "s", "cloudinary_folder": "studymind/documents"}, None,
     "Cloudinary folder is staging-specific"),
])
def test_each_unsafe_setting_fails_closed(over, env, expected):
    assert expected in fails(run(over, env))


def test_a_real_looking_key_is_the_only_thing_that_fails_when_it_is_set_with_a_proxy():
    # the proxy is configured, but the backend could still bypass it by holding the real key
    r = run({"openai_api_key": "sk-proj-" + "B" * 60})
    assert fails(r) == {"backend does NOT hold a real OpenAI key (proxy holds it)"}


def test_staging_cloudinary_folder_and_test_stripe_key_are_accepted():
    r = run({"cloudinary_cloud_name": "c", "cloudinary_api_key": "k", "cloudinary_api_secret": "s", "cloudinary_folder": "studymind-staging/documents",
             "studymind_billing_enabled": True, "studymind_stripe_secret_key": "sk_test_" + "a" * 30})
    assert fails(r) == set()


def test_missing_forbidden_hosts_is_called_out_not_silently_passed():
    r = run(forbid_host=[])
    assert any(level == "WARN" and "forbid-host" in name for level, name, _ in r)


def test_no_secret_value_is_ever_printed():
    secret = "S3cretValue" * 4
    r = run({"jwt_secret_key": secret, "typesense_api_key": secret, "openai_api_key": secret})
    assert secret not in " ".join(detail + name for _, name, detail in r)
