"""LLM client backed by any OpenAI-compatible gateway (e.g. LiteLLM).

Owns:
- structured output: a single tool call whose parameters schema is a Pydantic model.
- retries: tenacity exponential backoff on transient errors.

Note: Anthropic-style prompt caching is not forwarded — LiteLLM handles caching
at the gateway level based on its own configuration.
"""

from __future__ import annotations

import json
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


class LLMClient:
    def __init__(self, *, settings: Settings, sdk: Any | None = None) -> None:
        self._settings = settings
        self._sdk: Any = sdk or OpenAI(
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )

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
        response = self._sdk.chat.completions.create(
            model=self._settings.llm_model,
            max_tokens=max_tokens,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )

        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if tool_calls:
            payload = json.loads(tool_calls[0].function.arguments)
            return schema.model_validate(payload)

        raise StructuredOutputError(f"Model did not return a `{tool_name}` tool call.")


# Backwards-compatible alias
AnthropicClient = LLMClient
