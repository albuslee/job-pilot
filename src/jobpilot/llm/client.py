"""Thin Anthropic SDK wrapper.

Owns:
- prompt caching: system prompt + (optional) retrieved-context block both marked ephemeral.
- structured output: a single tool call whose input_schema is a Pydantic model.
- retries: tenacity exponential backoff on transient errors.
"""

from __future__ import annotations

from typing import Any, TypeVar

from anthropic import Anthropic, APIConnectionError, APIStatusError, RateLimitError
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


class AnthropicClient:
    def __init__(self, *, settings: Settings, sdk: Any | None = None) -> None:
        self._settings = settings
        self._sdk: Any = sdk or Anthropic(api_key=settings.anthropic_api_key)

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

        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        user_content: list[dict[str, Any]] = []
        if cached_context:
            user_content.append(
                {
                    "type": "text",
                    "text": cached_context,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        user_content.append({"type": "text", "text": user})

        tool = {
            "name": tool_name,
            "description": tool_description,
            "input_schema": schema.model_json_schema(),
        }

        log.debug("anthropic.request", model=self._settings.anthropic_model, tool=tool_name)
        response = self._sdk.messages.create(
            model=self._settings.anthropic_model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
        )

        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == tool_name
            ):
                payload = getattr(block, "input", {})
                return schema.model_validate(payload)

        raise StructuredOutputError(f"Model did not return a `{tool_name}` tool_use block.")
