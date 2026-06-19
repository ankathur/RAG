"""Shared test fixtures: fake LLM + embedder so tests need no live endpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import pytest

from rag.config import Settings
from rag.llm.base import LLMProvider
from rag.models import Document, Page


class FakeLLM(LLMProvider):
    """Deterministic stand-in for an OpenAI-compatible provider."""

    def __init__(
        self,
        responses: dict[str, dict] | None = None,
        default_sections: list[dict] | None = None,
        gen_text: str = "This is a grounded answer [S1].",
    ) -> None:
        self.model = "fake"
        self.responses = responses or {}
        self.default_sections = default_sections or []
        self.gen_text = gen_text

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        return self.gen_text

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        yield self.gen_text

    def structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str | None = None,
        schema_name: str = "result",
    ) -> dict[str, Any]:
        if schema_name in self.responses:
            return self.responses[schema_name]
        props = schema.get("properties", {})
        if "sections" in props:
            return {"sections": self.default_sections}
        if "selected_node_ids" in props:
            return {"selected_node_ids": []}
        if "ranking" in props:
            return {"ranking": []}
        return {}

    def ping(self) -> bool:
        return True


class FakeEmbedder:
    """Hashing bag-of-words embedder — deterministic, no model download."""

    dim = 32

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        chroma_dir=str(tmp_path / "chroma"),
        index_dir=str(tmp_path / "pageindex"),
        chunk_size=200,
        chunk_overlap=40,
    )


@pytest.fixture
def sample_doc() -> Document:
    return Document(
        id="handbook-test",
        source_path="handbook.md",
        format="md",
        pages=[
            Page(number=1, text="# Working Hours\nStandard hours are 9 to 17. "
                                "Annual leave is 20 days."),
            Page(number=2, text="# Remote Work\nRemote work up to three days per "
                                "week with approval. Core hours 10 to 15."),
            Page(number=3, text="# Security\nLaptops need encryption. Passwords "
                                "must be at least 14 characters."),
        ],
        metadata={"filename": "handbook.md"},
    )
