"""LLM client: factory and telemetry helpers for LangChain ChatOpenAI."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from jobpilot.config import Settings


@dataclass(frozen=True)
class CallTelemetry:
    """Per-LLM-call usage and latency."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def build_llm(settings: Settings) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at the LiteLLM gateway."""
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
        max_retries=3,
        max_tokens=2048,
    )


@contextmanager
def record(llm: ChatOpenAI) -> Iterator[list[CallTelemetry]]:
    """Context manager that captures CallTelemetry for every .invoke() call made
    on `llm` while the block is active.

    Intercepts calls on the `llm` instance itself. Wrap the block that drives
    graph execution — telemetry is captured whenever the graph calls `llm.invoke`
    internally::

        with record(llm) as calls:
            await graph.ainvoke(state)
        print(calls[0].input_tokens)
    """
    bucket: list[CallTelemetry] = []
    model_name = llm.model_name

    # Capture what's in the instance dict now (may be a prior mock, or absent).
    _missing = object()
    prior_instance_invoke = llm.__dict__.get("invoke", _missing)
    original_invoke = llm.invoke  # bound method or instance override — used to call through

    def _instrumented_invoke(input, config=None, **kwargs):  # type: ignore[no-untyped-def]
        t0 = time.monotonic()
        response = original_invoke(input, config, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000.0
        usage = getattr(response, "usage_metadata", None) or {}
        bucket.append(
            CallTelemetry(
                model=model_name,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                latency_ms=latency_ms,
            )
        )
        return response

    llm.__dict__["invoke"] = _instrumented_invoke  # type: ignore[index]
    try:
        yield bucket
    finally:
        if prior_instance_invoke is _missing:
            llm.__dict__.pop("invoke", None)  # type: ignore[union-attr]
        else:
            llm.__dict__["invoke"] = prior_instance_invoke  # type: ignore[index]
