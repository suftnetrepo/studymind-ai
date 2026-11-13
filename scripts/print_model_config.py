"""Prints the EFFECTIVE AI configuration of the environment it runs in, without secrets. Run it in staging.

    python scripts/print_model_config.py

Why: the code defaults (gpt-4.1-mini) can differ from what an environment actually sets (OPENAI_CHAT_MODEL). Cost
estimates, usage reservations and answer behaviour all depend on the model, so confirm it rather than assume it.
Exit code 1 if the chat model is missing from the price table (cost estimates would fall back to the priciest model).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "")          # never required to print the configuration
os.environ.setdefault("JWT_SECRET_KEY", "x" * 32)

from app.config import get_settings  # noqa: E402
from app.usage import policy as usage  # noqa: E402

s = get_settings()
price_in, price_out = usage.model_prices(s.openai_chat_model)
known = s.openai_chat_model in usage.MODEL_PRICES or any(s.openai_chat_model.startswith(f"{m}-") for m in usage.MODEL_PRICES)
print("chat model              :", s.openai_chat_model, "" if known else "  <-- NOT IN PRICE TABLE (priced as the most expensive model)")
print("chat price (USD/1M tok) : in", price_in, "/ out", price_out)
print("embedding model         :", s.openai_embedding_model, f"({s.openai_embedding_dimension} dims)")
print("OPENAI_API_KEY          :", "set" if os.environ.get("OPENAI_API_KEY") else "NOT set in this process (value never printed)")
print("OpenAI base URL override:", os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "none (api.openai.com)")
print("retries (llm / embed)   :", s.openai_llm_max_retries, "/", s.openai_embed_max_retries, "| timeout", s.openai_timeout_seconds, "s")
print("similarity threshold    :", s.similarity_threshold, "| top_k", s.top_k_retrieval, "| vector k", s.retrieval_vector_k)
print("chunk size / overlap    :", s.chunk_size, "/", s.chunk_overlap)
print("quota enforcement       :", "ON" if s.studymind_quota_enabled else "off")
print("worst-case reservation per request (micro-USD):", {f: usage.reservation_micro_usd(f, s.openai_chat_model) for f in usage.FEATURES})
sys.exit(0 if known else 1)
