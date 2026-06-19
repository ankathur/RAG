"""Build a retriever for a given mode (see SPEC.md §11)."""

from __future__ import annotations

from rag.config import RetrievalMode, Settings, get_settings
from rag.retrievers.base import Retriever


def build_retriever(mode: RetrievalMode, settings: Settings | None = None) -> Retriever:
    settings = settings or get_settings()
    if mode == "vector":
        from rag.retrievers.vector import VectorRetriever

        return VectorRetriever(settings)
    if mode == "pageindex":
        from rag.retrievers.pageindex import PageIndexRetriever

        return PageIndexRetriever(settings)
    if mode == "hybrid":
        from rag.retrievers.hybrid import HybridRetriever

        return HybridRetriever(settings)
    raise ValueError(f"Unknown retrieval mode: {mode!r}")
