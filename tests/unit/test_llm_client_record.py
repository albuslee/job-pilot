from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from jobpilot.config import Settings
from jobpilot.llm.client import CallTelemetry, build_llm, record


def _fake_llm(settings: Settings, *, in_tok: int = 100, out_tok: int = 20) -> MagicMock:
    """Return a real ChatOpenAI whose .invoke() is replaced with a mock that returns
    an AIMessage with usage_metadata."""
    llm = build_llm(settings)
    response = AIMessage(content="ok")
    response.usage_metadata = {"input_tokens": in_tok, "output_tokens": out_tok}  # type: ignore[assignment]
    # ChatOpenAI is a Pydantic model — direct attribute assignment is blocked.
    # Writing into __dict__ bypasses the guard while still shadowing the class method.
    llm.__dict__["invoke"] = MagicMock(return_value=response)  # type: ignore[index]
    return llm  # type: ignore[return-value]


def test_record_captures_token_counts_and_latency(settings: Settings) -> None:
    llm = _fake_llm(settings, in_tok=100, out_tok=20)

    with record(llm) as calls:
        llm.invoke("hello")

    assert len(calls) == 1
    t = calls[0]
    assert isinstance(t, CallTelemetry)
    assert t.model == settings.llm_model
    assert t.input_tokens == 100
    assert t.output_tokens == 20
    assert t.latency_ms >= 0


def test_record_accumulates_multiple_calls(settings: Settings) -> None:
    llm = build_llm(settings)
    r1 = AIMessage(content="a")
    r1.usage_metadata = {"input_tokens": 100, "output_tokens": 20}  # type: ignore[assignment]
    r2 = AIMessage(content="b")
    r2.usage_metadata = {"input_tokens": 200, "output_tokens": 30}  # type: ignore[assignment]
    llm.__dict__["invoke"] = MagicMock(side_effect=[r1, r2])  # type: ignore[index]

    with record(llm) as calls:
        llm.invoke("a")
        llm.invoke("b")

    assert [(c.input_tokens, c.output_tokens) for c in calls] == [(100, 20), (200, 30)]


def test_record_isolates_blocks(settings: Settings) -> None:
    llm = build_llm(settings)
    r1 = AIMessage(content="a")
    r1.usage_metadata = {"input_tokens": 100, "output_tokens": 20}  # type: ignore[assignment]
    r2 = AIMessage(content="b")
    r2.usage_metadata = {"input_tokens": 200, "output_tokens": 30}  # type: ignore[assignment]
    llm.__dict__["invoke"] = MagicMock(side_effect=[r1, r2])  # type: ignore[index]

    with record(llm) as a:
        llm.invoke("a")
    with record(llm) as b:
        llm.invoke("b")

    assert len(a) == 1 and len(b) == 1
    assert a[0].input_tokens == 100
    assert b[0].input_tokens == 200


def test_record_outside_block_does_not_record(settings: Settings) -> None:
    llm = _fake_llm(settings, in_tok=100, out_tok=20)

    llm.invoke("before block — should not be recorded")

    with record(llm) as calls:
        llm.invoke("inside block")

    assert len(calls) == 1


def test_record_restores_original_invoke_after_exception(settings: Settings) -> None:
    llm = _fake_llm(settings)
    original = llm.invoke

    with pytest.raises(ValueError, match="boom"), record(llm):
        raise ValueError("boom")

    assert llm.invoke is original


def test_record_restores_original_invoke_on_success(settings: Settings) -> None:
    llm = _fake_llm(settings)
    original = llm.invoke

    with record(llm):
        llm.invoke("hi")

    assert llm.invoke is original
