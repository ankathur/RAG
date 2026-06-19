"""Embedding provider interface + factory (see SPEC.md §7)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.config import Settings, get_settings


@runtime_checkable
class EmbeddingProvider(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    if settings.embedding_provider == "openai":
        from rag.embeddings.openai_compat import OpenAICompatEmbeddings

        return OpenAICompatEmbeddings(
            base_url=settings.embedding_base_url or settings.generation_base_url,
            api_key=settings.embedding_api_key or settings.generation_api_key,
            model=settings.embedding_model,
        )

    from rag.embeddings.local import LocalEmbeddings

    return LocalEmbeddings(model_name=settings.embedding_model)
