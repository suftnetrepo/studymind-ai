"""OpenAI client retry/timeout bounds are configurable and default to the previous library behaviour."""
import pytest

from app.agents import llm_factory
from app.config import get_settings


@pytest.fixture(autouse=True)
def _fresh():
    llm_factory.get_llm.cache_clear(); llm_factory.get_embed_model.cache_clear()
    yield
    llm_factory.get_llm.cache_clear(); llm_factory.get_embed_model.cache_clear()


def test_defaults_preserve_previous_behaviour():
    s = get_settings()
    assert (s.openai_llm_max_retries, s.openai_embed_max_retries, s.openai_timeout_seconds) == (3, 10, 60.0)
    assert llm_factory.get_llm().max_retries == 3 and llm_factory.get_embed_model().max_retries == 10


def test_bounds_are_applied_to_both_clients(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "openai_llm_max_retries", 1)
    monkeypatch.setattr(s, "openai_embed_max_retries", 0)
    monkeypatch.setattr(s, "openai_timeout_seconds", 20.0)
    llm, emb = llm_factory.get_llm(), llm_factory.get_embed_model()
    assert (llm.max_retries, llm.timeout) == (1, 20.0)
    assert (emb.max_retries, emb.timeout) == (0, 20.0)
    assert llm.max_tokens == 2048          # output cap used by the cost reservation must not change
