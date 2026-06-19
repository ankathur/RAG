"""Vector retrieval: chunk -> embed -> Chroma similarity search (SPEC.md §9.1)."""

from __future__ import annotations

from pathlib import Path

from rag.config import Settings, get_settings
from rag.embeddings.base import EmbeddingProvider, build_embedding_provider
from rag.models import Document, RetrievedContext
from rag.retrievers.base import Retriever, chunk_document

_COLLECTION = "rag_chunks"


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._collection = None

    # -- lazy resources ---------------------------------------------------------
    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = build_embedding_provider(self.settings)
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            Path(self.settings.chroma_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self.settings.chroma_dir)
            self._collection = client.get_or_create_collection(
                name=_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    # -- Retriever --------------------------------------------------------------
    def index(self, docs: list[Document]) -> None:
        collection = self._get_collection()
        for doc in docs:
            chunks = chunk_document(
                doc, self.settings.chunk_size, self.settings.chunk_overlap
            )
            if not chunks:
                continue
            embeddings = self.embedder.embed_documents([c.text for c in chunks])
            collection.upsert(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=embeddings,
                metadatas=[
                    {"doc_id": c.doc_id, "page": c.page or 0} for c in chunks
                ],
            )

    def retrieve(
        self, query: str, k: int, where: dict | None = None
    ) -> list[RetrievedContext]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        vec = self.embedder.embed_query(query)
        res = collection.query(
            query_embeddings=[vec],
            n_results=min(k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: list[RetrievedContext] = []
        for text, meta, dist in zip(docs, metas, dists):
            page = int(meta.get("page", 0)) if meta else 0
            out.append(
                RetrievedContext(
                    text=text,
                    doc_id=(meta or {}).get("doc_id", "unknown"),
                    locator=f"p.{page}" if page else "p.?",
                    score=round(max(0.0, 1.0 - float(dist)), 4),
                    origin="vector",
                )
            )
        return out
