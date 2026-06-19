"""Loader tests for Markdown and plain text (see SPEC.md §8)."""

from __future__ import annotations

from rag.documents.loader import load_document, load_documents


def test_markdown_splits_on_headings(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        "# Title\nIntro paragraph.\n\n## Section A\nAlpha.\n\n## Section B\nBeta.\n",
        encoding="utf-8",
    )
    doc = load_document(md)
    assert doc.format == "md"
    # Title block + Section A + Section B == 3 pseudo-pages.
    assert len(doc.pages) == 3
    assert "Section A" in doc.pages[1].text
    assert doc.pages[0].number == 1


def test_text_windows_large_file(tmp_path):
    txt = tmp_path / "big.txt"
    paragraph = ("lorem ipsum dolor sit amet " * 40).strip()
    txt.write_text("\n\n".join([paragraph] * 6), encoding="utf-8")
    doc = load_document(txt)
    assert doc.format == "txt"
    assert len(doc.pages) > 1  # exceeded the window, so multiple pseudo-pages


def test_doc_id_is_deterministic(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("same content", encoding="utf-8")
    id1 = load_document(p).id
    id2 = load_document(p).id
    assert id1 == id2
    assert id1.startswith("a-")


def test_load_documents_directory(tmp_path):
    (tmp_path / "one.md").write_text("# One\nbody", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two body", encoding="utf-8")
    docs = load_documents(str(tmp_path))
    assert {d.format for d in docs} == {"md", "txt"}
