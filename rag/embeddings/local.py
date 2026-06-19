"""Local, offline embeddings via sentence-transformers (see SPEC.md §7)."""

from __future__ import annotations


class LocalEmbeddings:
    """Wraps a sentence-transformers model. Loaded lazily on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        self._ensure()
        assert self._dim is not None
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
