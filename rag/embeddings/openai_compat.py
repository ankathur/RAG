"""Embeddings via any OpenAI-compatible /v1/embeddings endpoint (see SPEC.md §7)."""

from __future__ import annotations


class OpenAICompatEmbeddings:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
