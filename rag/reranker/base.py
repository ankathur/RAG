"""Reranker interface + factory (see SPEC.md §9.4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.config import Settings, get_settings
from rag.models import RetrievedContext


@runtime_checkable
class Reranker(Protocol):
    """Score and reorder retrieved contexts against a query."""

    def rerank(
        self, query: str, contexts: list[RetrievedContext], k: int
    ) -> list[RetrievedContext]:
        """Return the top-*k* contexts reranked by relevance to *query*.

        Implementations should update ``RetrievedContext.score`` with the
        reranker's relevance score and return contexts sorted best-first.
        """
        ...


def build_reranker(settings: Settings | None = None) -> Reranker:
    """Build a reranker from the current settings."""
    settings = settings or get_settings()
    if settings.reranker == "cross-encoder":
        from rag.reranker.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(model_name=settings.reranker_model)

    from rag.reranker.noop import NoOpReranker

    return NoOpReranker()
