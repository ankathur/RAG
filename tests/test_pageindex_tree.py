"""PageIndex tree-build and retrieval tests (see SPEC.md §9.2)."""

from __future__ import annotations

from rag.retrievers.pageindex import (
    PageIndexRetriever,
    _fallback_tree,
    _tree_from_sections,
)

from tests.conftest import FakeLLM


def test_tree_from_sections_nests_by_level():
    sections = [
        {"title": "Chapter 1", "summary": "c1", "start_page": 1, "end_page": 4, "level": 1},
        {"title": "1.1", "summary": "s11", "start_page": 1, "end_page": 2, "level": 2},
        {"title": "1.2", "summary": "s12", "start_page": 3, "end_page": 4, "level": 2},
        {"title": "Chapter 2", "summary": "c2", "start_page": 5, "end_page": 6, "level": 1},
    ]
    roots = _tree_from_sections(sections, page_count=6)
    assert len(roots) == 2
    assert roots[0].title == "Chapter 1"
    assert len(roots[0].nodes) == 2
    assert roots[0].nodes[0].title == "1.1"
    # node_ids are unique across the tree
    ids = [n.node_id for r in roots for n in r.walk()]
    assert len(ids) == len(set(ids)) == 4


def test_tree_from_sections_clamps_pages():
    sections = [{"title": "X", "start_page": 0, "end_page": 99, "level": 1}]
    roots = _tree_from_sections(sections, page_count=5)
    assert roots[0].start_page == 1
    assert roots[0].end_page == 5


def test_fallback_tree_covers_all_pages():
    roots = _fallback_tree(page_count=25, max_pages=10)
    assert [(r.start_page, r.end_page) for r in roots] == [(1, 10), (11, 20), (21, 25)]


def test_pageindex_build_and_retrieve(settings, sample_doc):
    llm = FakeLLM(
        default_sections=[
            {"title": "Working Hours", "summary": "hours and leave",
             "start_page": 1, "end_page": 1, "level": 1},
            {"title": "Remote Work", "summary": "remote policy",
             "start_page": 2, "end_page": 2, "level": 1},
            {"title": "Security", "summary": "encryption and passwords",
             "start_page": 3, "end_page": 3, "level": 1},
        ]
    )
    retriever = PageIndexRetriever(settings, llm=llm)
    retriever.index([sample_doc])

    # Persisted to disk.
    assert (tmp := settings.index_dir) and list_json(tmp)

    # LLM-driven selection: return a concrete node id.
    gid = f"{sample_doc.id}#0003"  # Security section
    llm.responses["node_selection"] = {"selected_node_ids": [gid]}
    contexts = retriever.retrieve("password length", k=3)
    assert contexts and contexts[0].origin == "pageindex"
    assert "14 characters" in contexts[0].text
    assert "Security" in contexts[0].locator


def test_pageindex_lexical_fallback(settings, sample_doc):
    llm = FakeLLM(
        default_sections=[
            {"title": "Remote Work", "summary": "remote policy and core hours",
             "start_page": 2, "end_page": 2, "level": 1},
        ]
    )
    retriever = PageIndexRetriever(settings, llm=llm)
    retriever.index([sample_doc])
    # node_selection returns [] -> falls back to lexical match on titles/summaries.
    contexts = retriever.retrieve("remote work approval", k=3)
    assert contexts
    assert any("Remote" in c.locator for c in contexts)


def test_pageindex_persistence_reload(settings, sample_doc):
    builder = PageIndexRetriever(
        settings,
        llm=FakeLLM(default_sections=[
            {"title": "Security", "summary": "encryption", "start_page": 3,
             "end_page": 3, "level": 1},
        ]),
    )
    builder.index([sample_doc])

    # Fresh instance loads trees from disk.
    reloaded = PageIndexRetriever(settings, llm=FakeLLM())
    reloaded.load()
    contexts = reloaded.retrieve("encryption", k=2)
    assert contexts and "encryption" in contexts[0].text.lower()


def list_json(directory: str) -> list[str]:
    from pathlib import Path

    return [str(p) for p in Path(directory).glob("*.json")]
