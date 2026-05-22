"""LLM client backed by any OpenAI-compatible gateway (e.g. LiteLLM).

Owns:
- structured output: a single tool call whose parameters schema is a Pydantic model.
- retries: tenacity exponential backoff on transient errors.

Note: Anthropic-style prompt caching is not forwarded — LiteLLM handles caching
at the gateway level based on its own configuration.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from jobpilot.config import Settings
from jobpilot.logging_setup import get_logger

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)


class StructuredOutputError(RuntimeError):
    """Raised when the model fails to emit the expected tool call."""


_RETRY_EXC = (APIConnectionError, RateLimitError, APIStatusError)


@dataclass(frozen=True)
class CallTelemetry:
    """Per-LLM-call usage and latency, captured by LLMClient.record()."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient:
    def __init__(self, *, settings: Settings, sdk: Any | None = None) -> None:
        self._settings = settings
        self._sdk: Any = sdk or OpenAI(
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        self._active_recording: list[CallTelemetry] | None = None

    @contextmanager
    def record(self) -> Iterator[list[CallTelemetry]]:
        """Capture per-call telemetry for the duration of the block.

        Nested entry is not supported in item 1 and raises RuntimeError.
        """
        if self._active_recording is not None:
            raise RuntimeError("LLMClient.record() blocks cannot be nested")
        bucket: list[CallTelemetry] = []
        self._active_recording = bucket
        try:
            yield bucket
        finally:
            self._active_recording = None

    @retry(
        retry=retry_if_exception_type(_RETRY_EXC),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        cached_context: str | None,
        schema: type[T],
        tool_name: str,
        tool_description: str,
        max_tokens: int = 2048,
    ) -> T:
        """Force the model to emit a single tool call whose input parses into `schema`."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if cached_context:
            messages.append({"role": "user", "content": cached_context})
            messages.append({"role": "assistant", "content": "Understood."})
        messages.append({"role": "user", "content": user})

        tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": schema.model_json_schema(),
            },
        }

        log.debug("llm.request", model=self._settings.llm_model, tool=tool_name)
        t0 = time.monotonic()
        response = self._sdk.chat.completions.create(
            model=self._settings.llm_model,
            max_tokens=max_tokens,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        latency_ms = (time.monotonic() - t0) * 1000.0

        if self._active_recording is not None:
            usage = getattr(response, "usage", None)
            self._active_recording.append(
                CallTelemetry(
                    model=self._settings.llm_model,
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_ms=latency_ms,
                )
            )

        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if tool_calls:
            payload = json.loads(tool_calls[0].function.arguments)
            return schema.model_validate(payload)

        raise StructuredOutputError(f"Model did not return a `{tool_name}` tool call.")


# Backwards-compatible alias
AnthropicClient = LLMClient
