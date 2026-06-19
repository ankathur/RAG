"""End-to-end RAG pipeline: ingest + ask (see SPEC.md §11)."""

from __future__ import annotations

from rag.config import RetrievalMode, Settings, get_settings
from rag.documents.loader import load_documents
from rag.factory import build_retriever
from rag.generation.generator import Generator
from rag.models import Answer
from rag.retrievers.base import Retriever


class RAGPipeline:
    """Owns the retrievers (one per mode, lazily built) and the generator.

    Ingestion always builds the vector *and* PageIndex indexes so any mode —
    including hybrid — can answer afterwards.
    """

    def __init__(
        self, settings: Settings | None = None, generator: Generator | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.generator = generator or Generator(self.settings)
        self._retrievers: dict[str, Retriever] = {}
        self._loaded: set[str] = set()

    def _get(self, mode: RetrievalMode) -> Retriever:
        if mode not in self._retrievers:
            self._retrievers[mode] = build_retriever(mode, self.settings)
        retriever = self._retrievers[mode]
        if mode not in self._loaded:
            retriever.load()
            self._loaded.add(mode)
        return retriever

    def ingest(self, paths: list[str] | str) -> dict:
        docs = load_documents(paths)
        # Build both underlying indexes so every mode is queryable.
        self._get("vector").index(docs)
        self._get("pageindex").index(docs)
        # Drop any cached hybrid so it reloads the fresh indexes on next use.
        self._retrievers.pop("hybrid", None)
        self._loaded.discard("hybrid")
        return {
            "ingested": [d.id for d in docs],
            "vector": True,
            "pageindex": True,
        }

    def ask(
        self, query: str, mode: RetrievalMode | None = None, top_k: int | None = None
    ) -> Answer:
        mode = mode or self.settings.retrieval_mode
        k = top_k or self.settings.top_k
        retriever = self._get(mode)
        contexts = retriever.retrieve(query, k)
        return self.generator.answer(query, contexts, mode)
