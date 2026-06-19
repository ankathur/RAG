"""Common retriever interface + shared chunking (see SPEC.md §9)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.models import Chunk, Document, RetrievedContext


class Retriever(ABC):
    """One retrieval strategy. All modes share this interface so the pipeline
    and API are mode-agnostic."""

    name: str = "base"

    @abstractmethod
    def index(self, docs: list[Document]) -> None:
        """Build and persist the index for ``docs`` (idempotent per doc)."""

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievedContext]:
        """Return up to ``k`` relevant contexts for ``query``."""

    def persist(self) -> None:  # default: index() persists eagerly
        return None

    def load(self) -> None:  # default: nothing to restore
        return None


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split a document into overlapping character chunks, page-aware."""
    chunks: list[Chunk] = []
    position = 0
    for page in doc.pages:
        text = page.text
        if not text.strip():
            continue
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        id=f"{doc.id}:{page.number}:{position}",
                        doc_id=doc.id,
                        text=piece,
                        page=page.number,
                        position=position,
                    )
                )
                position += 1
            if end >= n:
                break
            start = end - overlap if end - overlap > start else end
    return chunks
