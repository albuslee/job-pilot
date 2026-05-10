from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from jobpilot.config import Settings
from jobpilot.llm.client import AnthropicClient
from jobpilot.models.schemas import EvaluationResult


class _FakeToolUseBlock:
    def __init__(self, name: str, payload: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = payload


class _FakeMessage:
    def __init__(self, blocks: list[Any]) -> None:
        self.content = blocks


def test_client_requests_structured_output_via_tool(settings: Settings) -> None:
    fake_payload = {
        "score": 78,
        "decision": "apply",
        "reasoning": "Solid backend match.",
        "cited_chunk_ids": ["cv-001"],
        "risk_flags": [],
    }
    sdk = MagicMock()
    sdk.messages.create.return_value = _FakeMessage(
        [_FakeToolUseBlock("submit_evaluation", fake_payload)]
    )
    client = AnthropicClient(settings=settings, sdk=sdk)

    result = client.complete_structured(
        system="system prompt",
        user="user prompt",
        cached_context="profile context to cache",
        schema=EvaluationResult,
        tool_name="submit_evaluation",
        tool_description="Submit your evaluation.",
    )

    assert isinstance(result, EvaluationResult)
    assert result.score == 78
    assert result.decision == "apply"

    call_kwargs = sdk.messages.create.call_args.kwargs
    assert call_kwargs["model"] == settings.anthropic_model
    # System is sent as a list of blocks with cache_control on the system prompt.
    system_blocks = call_kwargs["system"]
    assert any(
        isinstance(b, dict) and b.get("cache_control", {}).get("type") == "ephemeral"
        for b in system_blocks
    )
    # Cached context should appear as the first user content block, also cached.
    user_msg = call_kwargs["messages"][0]
    assert user_msg["role"] == "user"
    cached_block = user_msg["content"][0]
    assert cached_block["text"] == "profile context to cache"
    assert cached_block["cache_control"] == {"type": "ephemeral"}
    # tool_choice forces the model into the right tool.
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "submit_evaluation"}


def test_client_skips_cache_block_when_no_cached_context(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.messages.create.return_value = _FakeMessage(
        [
            _FakeToolUseBlock(
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
    client = AnthropicClient(settings=settings, sdk=sdk)
    client.complete_structured(
        system="s",
        user="u",
        cached_context=None,
        schema=EvaluationResult,
        tool_name="submit_evaluation",
        tool_description="d",
    )
    user_msg = sdk.messages.create.call_args.kwargs["messages"][0]
    assert len(user_msg["content"]) == 1
    assert user_msg["content"][0]["text"] == "u"
