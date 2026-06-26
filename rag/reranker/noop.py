"""No-op reranker — pass-through that preserves the original ordering."""

from __future__ import annotations

from rag.models import RetrievedContext


class NoOpReranker:
    """Returns contexts unchanged; used when reranking is disabled."""

    def rerank(
        self, query: str, contexts: list[RetrievedContext], k: int
    ) -> list[RetrievedContext]:
        return contexts[:k]
