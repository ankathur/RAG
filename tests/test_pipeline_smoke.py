"""End-to-end smoke tests with a mocked LLM (see SPEC.md §17)."""

from __future__ import annotations

import pytest

from rag.generation.generator import _NO_CONTEXT, Generator
from rag.models import RetrievedContext
from rag.retrievers.pageindex import PageIndexRetriever

from tests.conftest import FakeEmbedder, FakeLLM

_SECTIONS = [
    {"title": "Working Hours", "summary": "hours and annual leave",
     "start_page": 1, "end_page": 1, "level": 1},
    {"title": "Remote Work", "summary": "remote policy",
     "start_page": 2, "end_page": 2, "level": 1},
    {"title": "Security", "summary": "encryption and passwords",
     "start_page": 3, "end_page": 3, "level": 1},
]


def test_generator_grounded_answer(settings):
    gen = Generator(settings, llm=FakeLLM(gen_text="Leave is 20 days [S1]."))
    contexts = [
        RetrievedContext(text="Annual leave is 20 days.", doc_id="d", locator="p.1")
    ]
    ans = gen.answer("how much leave?", contexts, "pageindex")
    assert ans.text == "Leave is 20 days [S1]."
    assert len(ans.citations) == 1
    assert ans.mode == "pageindex"


def test_generator_no_context(settings):
    gen = Generator(settings, llm=FakeLLM())
    ans = gen.answer("anything", [], "vector")
    assert ans.text == _NO_CONTEXT
    assert ans.citations == []


def test_end_to_end_pageindex(settings, sample_doc):
    llm = FakeLLM(default_sections=_SECTIONS, gen_text="Answer grounded in [S1].")
    retriever = PageIndexRetriever(settings, llm=llm)
    retriever.index([sample_doc])
    contexts = retriever.retrieve("annual leave days", k=3)
    assert contexts

    gen = Generator(settings, llm=llm)
    ans = gen.answer("annual leave days", contexts, "pageindex")
    assert ans.text == "Answer grounded in [S1]."
    assert ans.citations


def test_pipeline_all_modes(settings, monkeypatch):
    pytest.importorskip("chromadb")

    fake = FakeLLM(default_sections=_SECTIONS, gen_text="Grounded answer [S1].")
    # Replace network-bound providers everywhere they are used.
    monkeypatch.setattr("rag.generation.generator.build_llm", lambda role, s=None: fake)
    monkeypatch.setattr("rag.retrievers.pageindex.build_llm", lambda role, s=None: fake)
    monkeypatch.setattr("rag.retrievers.hybrid.build_llm", lambda role, s=None: fake)
    monkeypatch.setattr(
        "rag.retrievers.vector.build_embedding_provider", lambda s=None: FakeEmbedder()
    )

    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(settings)
    result = pipeline.ingest("data/sample/handbook.md")
    assert result["vector"] and result["pageindex"]
    assert result["ingested"]

    for mode in ("vector", "pageindex", "hybrid"):
        ans = pipeline.ask("How much annual leave do employees get?", mode=mode)
        assert ans.mode == mode
        # Either grounded answer or the explicit no-context message — never a crash.
        assert ans.text
