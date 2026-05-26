from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from jobpilot.config import Settings
from jobpilot.llm.client import CallTelemetry, LLMClient
from jobpilot.models.schemas import EvaluationResult


def _fake_response(payload: dict[str, Any], in_tok: int, out_tok: int) -> MagicMock:
    """Mirror the existing test fakes in test_llm_client.py but include usage."""
    response = MagicMock()
    tool_call = MagicMock()
    tool_call.function.name = "submit_evaluation"
    tool_call.function.arguments = json.dumps(payload)
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [tool_call]
    response.usage.prompt_tokens = in_tok
    response.usage.completion_tokens = out_tok
    return response


def _payload() -> dict[str, Any]:
    return {
        "score": 50,
        "decision": "maybe",
        "reasoning": "ok",
        "cited_chunk_ids": [],
        "risk_flags": [],
    }


def test_record_captures_token_counts_and_latency(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _fake_response(_payload(), 100, 20)
    client = LLMClient(settings=settings, sdk=sdk)

    with client.record() as calls:
        client.complete_structured(
            system="s",
            user="u",
            cached_context=None,
            schema=EvaluationResult,
            tool_name="submit_evaluation",
            tool_description="d",
        )

    assert len(calls) == 1
    t = calls[0]
    assert isinstance(t, CallTelemetry)
    assert t.model == settings.llm_model
    assert t.input_tokens == 100
    assert t.output_tokens == 20
    assert t.latency_ms >= 0  # real wall time


def test_record_accumulates_multiple_calls(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = [
        _fake_response(_payload(), 100, 20),
        _fake_response(_payload(), 200, 30),
    ]
    client = LLMClient(settings=settings, sdk=sdk)

    with client.record() as calls:
        for _ in range(2):
            client.complete_structured(
                system="s",
                user="u",
                cached_context=None,
                schema=EvaluationResult,
                tool_name="submit_evaluation",
                tool_description="d",
            )

    assert [(c.input_tokens, c.output_tokens) for c in calls] == [(100, 20), (200, 30)]


def test_record_isolates_blocks(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = [
        _fake_response(_payload(), 100, 20),
        _fake_response(_payload(), 200, 30),
    ]
    client = LLMClient(settings=settings, sdk=sdk)

    with client.record() as a:
        client.complete_structured(
            system="s",
            user="u",
            cached_context=None,
            schema=EvaluationResult,
            tool_name="submit_evaluation",
            tool_description="d",
        )
    with client.record() as b:
        client.complete_structured(
            system="s",
            user="u",
            cached_context=None,
            schema=EvaluationResult,
            tool_name="submit_evaluation",
            tool_description="d",
        )

    assert len(a) == 1 and len(b) == 1
    assert a[0].input_tokens == 100
    assert b[0].input_tokens == 200


def test_record_outside_block_does_not_record(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _fake_response(_payload(), 100, 20)
    client = LLMClient(settings=settings, sdk=sdk)
    # No exception, and no recording state lingers.
    client.complete_structured(
        system="s",
        user="u",
        cached_context=None,
        schema=EvaluationResult,
        tool_name="submit_evaluation",
        tool_description="d",
    )
    with client.record() as calls:
        client.complete_structured(
            system="s",
            user="u",
            cached_context=None,
            schema=EvaluationResult,
            tool_name="submit_evaluation",
            tool_description="d",
        )
    assert len(calls) == 1  # not 2 — first call was before record() started


def test_record_rejects_nested_entry(settings: Settings) -> None:
    sdk = MagicMock()
    client = LLMClient(settings=settings, sdk=sdk)
    with client.record():  # noqa: SIM117
        with pytest.raises(RuntimeError, match="nested"):
            with client.record():
                pass


def test_record_resets_state_after_exception(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _fake_response(_payload(), 100, 20)
    client = LLMClient(settings=settings, sdk=sdk)

    # First block raises mid-way; the finally must reset state.
    with pytest.raises(ValueError, match="boom"):  # noqa: SIM117
        with client.record():
            raise ValueError("boom")

    # Second block must work, proving state was cleaned up.
    with client.record() as calls:
        client.complete_structured(
            system="s",
            user="u",
            cached_context=None,
            schema=EvaluationResult,
            tool_name="submit_evaluation",
            tool_description="d",
        )
    assert len(calls) == 1
