"""
Capture OpenAI token usage for the current request — only while a usage-policy guard is active.

`report(response)` is called after each LLM completion. With no active capture (every request
that isn't a policy-limited pass session) it does nothing, so behaviour is unchanged.
Context variables propagate into run_in_threadpool, so captures work for the sync pipelines.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

_capture: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar("llm_usage_capture", default=None)


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def extract_usage(response: Any) -> dict | None:
    """Token counts from a LlamaIndex completion or raw OpenAI response, or None."""
    raw = _get(response, "raw") or response
    usage = _get(raw, "usage")
    if usage is None:
        return None
    input_tokens, output_tokens = _get(usage, "prompt_tokens"), _get(usage, "completion_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    return {"input_tokens": int(input_tokens), "output_tokens": int(output_tokens), "model": _get(raw, "model")}


def report(response: Any) -> None:
    """Record one completion's usage into the active capture. Never raises."""
    sink = _capture.get()
    if sink is None:
        return
    try:
        sink.append(extract_usage(response) or {"missing": True})
    except Exception:
        sink.append({"missing": True})


@contextmanager
def capture() -> Iterator[list[dict]]:
    """Collect usage of every LLM call made inside the block (including in worker threads)."""
    sink: list[dict] = []
    token = _capture.set(sink)
    try:
        yield sink
    finally:
        _capture.reset(token)
