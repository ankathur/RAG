"""Tests for the reranker module."""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.models import RetrievedContext
from rag.reranker.base import build_reranker
from rag.reranker.noop import NoOpReranker


def _ctx(text: str, score: float | None = None, origin: str = "vector") -> RetrievedContext:
    return RetrievedContext(
        text=text, doc_id="doc1", locator="p.1", score=score, origin=origin
    )


# -- NoOpReranker -----------------------------------------------------------

def test_noop_reranker_passthrough():
    reranker = NoOpReranker()
    contexts = [_ctx("a", 0.9), _ctx("b", 0.5), _ctx("c", 0.3)]
    result = reranker.rerank("query", contexts, k=3)
    assert result == contexts


def test_noop_reranker_truncates():
    reranker = NoOpReranker()
    contexts = [_ctx("a"), _ctx("b"), _ctx("c"), _ctx("d")]
    result = reranker.rerank("query", contexts, k=2)
    assert len(result) == 2
    assert result[0].text == "a"
    assert result[1].text == "b"


def test_noop_reranker_empty():
    reranker = NoOpReranker()
    result = reranker.rerank("query", [], k=5)
    assert result == []


# -- build_reranker factory --------------------------------------------------

def test_build_reranker_none(tmp_path):
    settings = Settings(
        reranker="none",
        chroma_dir=str(tmp_path / "chroma"),
        index_dir=str(tmp_path / "pi"),
    )
    reranker = build_reranker(settings)
    assert isinstance(reranker, NoOpReranker)


def test_build_reranker_cross_encoder(tmp_path):
    settings = Settings(
        reranker="cross-encoder",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        chroma_dir=str(tmp_path / "chroma"),
        index_dir=str(tmp_path / "pi"),
    )
    reranker = build_reranker(settings)
    from rag.reranker.cross_encoder import CrossEncoderReranker
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


# -- CrossEncoderReranker (with mock) ----------------------------------------

class FakeCrossEncoderModel:
    """Stand-in for sentence_transformers.CrossEncoder.predict()."""

    def predict(self, pairs):
        # Score inversely by text length — shorter passages rank higher.
        return [1.0 / (1 + len(p[1])) for p in pairs]


def test_cross_encoder_reranker_reorders():
    from rag.reranker.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="fake")
    reranker._model = FakeCrossEncoderModel()  # inject mock

    contexts = [
        _ctx("a long passage with many words"),
        _ctx("short"),
        _ctx("medium text here"),
    ]
    result = reranker.rerank("anything", contexts, k=2)

    assert len(result) == 2
    # "short" has highest score (shortest), should be first
    assert result[0].text == "short"
    # All returned contexts should have scores set
    assert all(c.score is not None for c in result)
    # Scores should be descending
    assert result[0].score >= result[1].score


def test_cross_encoder_reranker_empty():
    from rag.reranker.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="fake")
    reranker._model = FakeCrossEncoderModel()

    result = reranker.rerank("anything", [], k=5)
    assert result == []


def test_cross_encoder_reranker_single():
    from rag.reranker.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="fake")
    reranker._model = FakeCrossEncoderModel()

    contexts = [_ctx("only one")]
    result = reranker.rerank("anything", contexts, k=3)
    assert len(result) == 1
    assert result[0].text == "only one"


def test_cross_encoder_reranker_scores_overwrite_originals():
    from rag.reranker.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="fake")
    reranker._model = FakeCrossEncoderModel()

    contexts = [
        _ctx("hello world", score=0.99, origin="vector"),
        _ctx("hi", score=None, origin="pageindex"),
    ]
    result = reranker.rerank("query", contexts, k=2)
    # Both contexts should have new scores from the reranker
    for ctx in result:
        assert ctx.score is not None
    # "hi" is shorter so it should score higher
    assert result[0].text == "hi"


# -- Pipeline integration with reranker -------------------------------------

def test_pipeline_ask_with_noop_reranker(settings, sample_doc, monkeypatch):
    """Pipeline works end-to-end with the no-op reranker (default)."""
    pytest.importorskip("chromadb")

    from tests.conftest import FakeEmbedder, FakeLLM

    _SECTIONS = [
        {"title": "Working Hours", "summary": "hours and annual leave",
         "start_page": 1, "end_page": 1, "level": 1},
    ]
    fake = FakeLLM(default_sections=_SECTIONS, gen_text="Grounded answer [S1].")
    monkeypatch.setattr("rag.generation.generator.build_llm", lambda role, s=None: fake)
    monkeypatch.setattr("rag.retrievers.pageindex.build_llm", lambda role, s=None: fake)
    monkeypatch.setattr("rag.retrievers.hybrid.build_llm", lambda role, s=None: fake)
    monkeypatch.setattr(
        "rag.retrievers.vector.build_embedding_provider", lambda s=None: FakeEmbedder()
    )

    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(settings)
    pipeline.ingest("data/sample/handbook.md")
    ans = pipeline.ask("annual leave", mode="vector")
    assert ans.text
    assert ans.mode == "vector"


def test_pipeline_ask_with_cross_encoder_reranker(settings, sample_doc, monkeypatch):
    """Pipeline works with a (mock) cross-encoder reranker."""
    pytest.importorskip("chromadb")

    from tests.conftest import FakeEmbedder, FakeLLM
    from rag.reranker.cross_encoder import CrossEncoderReranker

    _SECTIONS = [
        {"title": "Working Hours", "summary": "hours and annual leave",
         "start_page": 1, "end_page": 1, "level": 1},
    ]
    fake = FakeLLM(default_sections=_SECTIONS, gen_text="Grounded answer [S1].")
    monkeypatch.setattr("rag.generation.generator.build_llm", lambda role, s=None: fake)
    monkeypatch.setattr("rag.retrievers.pageindex.build_llm", lambda role, s=None: fake)
    monkeypatch.setattr("rag.retrievers.hybrid.build_llm", lambda role, s=None: fake)
    monkeypatch.setattr(
        "rag.retrievers.vector.build_embedding_provider", lambda s=None: FakeEmbedder()
    )

    # Enable the cross-encoder reranker with a fake model
    settings.reranker = "cross-encoder"
    reranker = CrossEncoderReranker(model_name="fake")
    reranker._model = FakeCrossEncoderModel()

    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(settings, reranker=reranker)
    pipeline.ingest("data/sample/handbook.md")
    ans = pipeline.ask("annual leave", mode="vector")
    assert ans.text
    assert ans.mode == "vector"
    # With the cross-encoder, citations should have scores
    for c in ans.citations:
        assert c.score is not None
