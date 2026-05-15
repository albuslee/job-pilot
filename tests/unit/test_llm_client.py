from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from jobpilot.config import Settings
from jobpilot.llm.client import LLMClient, AnthropicClient  # AnthropicClient is an alias
from jobpilot.models.schemas import EvaluationResult


class _FakeFunction:
    def __init__(self, name: str, payload: dict[str, Any]) -> None:
        self.name = name
        self.arguments = json.dumps(payload)


class _FakeToolCall:
    def __init__(self, name: str, payload: dict[str, Any]) -> None:
        self.function = _FakeFunction(name, payload)


class _FakeMessage:
    def __init__(self, tool_calls: list[_FakeToolCall]) -> None:
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, tool_calls: list[_FakeToolCall]) -> None:
        self.message = _FakeMessage(tool_calls)


class _FakeResponse:
    def __init__(self, tool_calls: list[_FakeToolCall]) -> None:
        self.choices = [_FakeChoice(tool_calls)]


def test_client_requests_structured_output_via_tool(settings: Settings) -> None:
    fake_payload = {
        "score": 78,
        "decision": "apply",
        "reasoning": "Solid backend match.",
        "cited_chunk_ids": ["cv-001"],
        "risk_flags": [],
    }
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _FakeResponse(
        [_FakeToolCall("submit_evaluation", fake_payload)]
    )
    client = LLMClient(settings=settings, sdk=sdk)

    result = client.complete_structured(
        system="system prompt",
        user="user prompt",
        cached_context="profile context",
        schema=EvaluationResult,
        tool_name="submit_evaluation",
        tool_description="Submit your evaluation.",
    )

    assert isinstance(result, EvaluationResult)
    assert result.score == 78
    assert result.decision == "apply"

    call_kwargs = sdk.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == settings.llm_model
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "system prompt"}
    assert messages[1] == {"role": "user", "content": "profile context"}
    assert messages[-1] == {"role": "user", "content": "user prompt"}
    tool = call_kwargs["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "submit_evaluation"
    assert call_kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit_evaluation"}}


def test_client_skips_context_messages_when_no_cached_context(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _FakeResponse(
        [
            _FakeToolCall(
                "submit_evaluation",
                {
                    "score": 10,
                    "decision": "skip",
                    "reasoning": "no",
                    "cited_chunk_ids": [],
                    "risk_flags": [],
                },
            )
        ]
    )
    client = LLMClient(settings=settings, sdk=sdk)
    client.complete_structured(
        system="s",
        user="u",
        cached_context=None,
        schema=EvaluationResult,
        tool_name="submit_evaluation",
        tool_description="d",
    )
    messages = sdk.chat.completions.create.call_args.kwargs["messages"]
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "s"}
    assert messages[1] == {"role": "user", "content": "u"}


def test_anthropic_client_alias(settings: Settings) -> None:
    assert AnthropicClient is LLMClient
