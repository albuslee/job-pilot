"""Embedding provider: voyage (default) or openai. Selected by Settings.embedding_provider."""

from __future__ import annotations

from typing import Any, Protocol

from jobpilot.config import Settings


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class _VoyageEmbedder:
    def __init__(self, sdk: Any, model: str) -> None:
        self._sdk = sdk
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._sdk.embed(texts, model=self._model, input_type="document")
        return [list(v) for v in result.embeddings]


class _OpenAIEmbedder:
    def __init__(self, sdk: Any, model: str) -> None:
        self._sdk = sdk
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._sdk.embeddings.create(model=self._model, input=texts)
        return [list(item.embedding) for item in result.data]


class _OllamaEmbedder:
    def __init__(self, sdk: Any, model: str) -> None:
        self._sdk = sdk
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._sdk.embeddings.create(model=self._model, input=texts)
        return [list(item.embedding) for item in result.data]


def build_embedder(
    *,
    settings: Settings,
    voyage_sdk: Any | None = None,
    openai_sdk: Any | None = None,
    ollama_sdk: Any | None = None,
) -> Embedder:
    if settings.embedding_provider == "ollama":
        if ollama_sdk is None:
            from openai import OpenAI

            ollama_sdk = OpenAI(base_url=f"{settings.ollama_base_url}/v1", api_key="ollama")
        return _OllamaEmbedder(ollama_sdk, settings.ollama_embedding_model)

    if settings.embedding_provider == "voyage":
        if voyage_sdk is None:
            import voyageai

            if not settings.voyage_api_key:
                raise ValueError("VOYAGE_API_KEY is required when EMBEDDING_PROVIDER=voyage")
            voyage_sdk = voyageai.Client(api_key=settings.voyage_api_key)  # type: ignore[attr-defined]
        return _VoyageEmbedder(voyage_sdk, settings.voyage_model)

    if openai_sdk is None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        openai_sdk = OpenAI(api_key=settings.openai_api_key)
    return _OpenAIEmbedder(openai_sdk, settings.openai_embedding_model)
