"""
LLM and embedding model factory — standard OpenAI API.
Returns configured OpenAI instances ready for LlamaIndex.
Both are cached as module-level singletons.
"""
from __future__ import annotations

from functools import lru_cache

from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from app.config import get_settings
from app.logging_config import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> OpenAI:
    s = get_settings()
    log.info("llm_init", model=s.openai_chat_model)
    return OpenAI(
        model=s.openai_chat_model,
        api_key=s.openai_api_key,
        temperature=0.1,
        max_tokens=2048,
    )


@lru_cache(maxsize=1)
def get_embed_model() -> OpenAIEmbedding:
    s = get_settings()
    log.info("embed_model_init", model=s.openai_embedding_model)
    return OpenAIEmbedding(
        model=s.openai_embedding_model,
        api_key=s.openai_api_key,
        embed_batch_size=32,
        dimensions=s.openai_embedding_dimension,
    )


def configure_llama_settings() -> None:
    """Push LLM + embed model into LlamaIndex global Settings."""
    LlamaSettings.llm          = get_llm()
    LlamaSettings.embed_model  = get_embed_model()
    LlamaSettings.chunk_size   = get_settings().chunk_size
    LlamaSettings.chunk_overlap = get_settings().chunk_overlap
    log.info("llama_settings_configured")
