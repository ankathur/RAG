"""Cross-encoder reranker via sentence-transformers (see SPEC.md §9.4)."""

from __future__ import annotations

import logging

from rag.models import RetrievedContext

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Wraps a sentence-transformers ``CrossEncoder`` model.

    The model is loaded lazily on the first ``rerank()`` call so that startup
    cost is deferred until the reranker is actually used.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("Loading cross-encoder model %s …", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, contexts: list[RetrievedContext], k: int
    ) -> list[RetrievedContext]:
        if not contexts:
            return []
        if len(contexts) <= 1:
            return contexts[:k]

        model = self._ensure()
        pairs = [[query, ctx.text] for ctx in contexts]
        scores = model.predict(pairs)

        scored = []
        for ctx, score in zip(contexts, scores):
            reranked = ctx.model_copy(update={"score": round(float(score), 4)})
            scored.append(reranked)

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]
