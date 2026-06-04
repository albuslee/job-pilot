from __future__ import annotations

from jobpilot.config import Settings
from jobpilot.llm.client import build_llm


def test_build_llm_returns_chat_openai_with_gateway_settings(settings: Settings) -> None:
    from langchain_openai import ChatOpenAI

    llm = build_llm(settings)

    assert isinstance(llm, ChatOpenAI)
    # openai_api_base is how langchain_openai exposes the base_url
    assert str(llm.openai_api_base) == settings.litellm_base_url
    assert llm.model_name == settings.llm_model
    assert llm.max_retries == 3


def test_build_llm_returns_new_instance_each_call(settings: Settings) -> None:
    llm_a = build_llm(settings)
    llm_b = build_llm(settings)
    assert llm_a is not llm_b
