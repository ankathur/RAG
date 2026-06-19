"""Document ingestion for PDF, Markdown, and plain text (see SPEC.md §8)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from rag.models import Document, Page

SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt", ".text"}
_TXT_WINDOW_CHARS = 3000
_HEADING_RE = re.compile(r"^#{1,6}\s+\S")


def load_documents(paths: list[str] | str) -> list[Document]:
    """Load every supported file in ``paths`` (files and/or directories)."""
    if isinstance(paths, str):
        paths = [paths]
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(
                sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTS)
            )
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"Path not found: {raw}")

    docs: list[Document] = []
    for f in files:
        ext = f.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED_EXTS)}"
            )
        docs.append(load_document(f))
    return docs


def load_document(path: str | Path) -> Document:
    path = Path(path)
    ext = path.suffix.lower()
    raw_bytes = path.read_bytes()
    doc_id = _doc_id(path, raw_bytes)

    if ext == ".pdf":
        pages, metadata = _load_pdf(path)
        fmt = "pdf"
    else:
        text = raw_bytes.decode("utf-8", errors="replace")
        if ext in (".md", ".markdown"):
            pages, metadata = _load_markdown(text)
            fmt = "md"
        else:
            pages, metadata = _load_text(text)
            fmt = "txt"

    metadata["filename"] = path.name
    return Document(
        id=doc_id, source_path=str(path), format=fmt, pages=pages, metadata=metadata
    )


def _doc_id(path: Path, content: bytes) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem).strip("-") or "doc"
    digest = hashlib.sha1(content).hexdigest()[:8]
    return f"{stem}-{digest}"


# --- PDF -----------------------------------------------------------------------
def _load_pdf(path: Path) -> tuple[list[Page], dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [
        Page(number=i + 1, text=(page.extract_text() or "").strip())
        for i, page in enumerate(reader.pages)
    ]
    metadata: dict = {"page_count": len(pages)}
    outline = _extract_outline(reader)
    if outline:
        metadata["outline"] = outline  # [{title, page}] — 1-based pages
    return pages, metadata


def _extract_outline(reader) -> list[dict]:
    """Flatten a PDF's bookmark outline into [{title, page}] (1-based)."""
    try:
        raw = reader.outline
    except Exception:
        return []

    flat: list[dict] = []

    def _walk(items):
        for item in items:
            if isinstance(item, list):
                _walk(item)
                continue
            try:
                title = str(getattr(item, "title", "")).strip()
                page_index = reader.get_destination_page_number(item)
            except Exception:
                continue
            if title and page_index is not None:
                flat.append({"title": title, "page": int(page_index) + 1})

    try:
        _walk(raw)
    except Exception:
        return []
    return flat


# --- Markdown ------------------------------------------------------------------
def _load_markdown(text: str) -> tuple[list[Page], dict]:
    """Split a Markdown doc into pseudo-pages at top-of-section headings."""
    lines = text.splitlines()
    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if _HEADING_RE.match(line) and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    if not sections:
        sections = [lines]

    pages = [
        Page(number=i + 1, text="\n".join(sec).strip())
        for i, sec in enumerate(sections)
        if "".join(sec).strip()
    ]
    if not pages:  # empty file -> single empty page
        pages = [Page(number=1, text="")]
    return pages, {"page_count": len(pages), "split": "markdown-heading"}


# --- Plain text ----------------------------------------------------------------
def _load_text(text: str) -> tuple[list[Page], dict]:
    """Window plain text into pseudo-pages on paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    pages: list[Page] = []
    buf = ""
    number = 1
    for para in paragraphs:
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) > _TXT_WINDOW_CHARS and buf:
            pages.append(Page(number=number, text=buf.strip()))
            number += 1
            buf = para
        else:
            buf = candidate
    if buf.strip() or not pages:
        pages.append(Page(number=number, text=buf.strip()))
    return pages, {"page_count": len(pages), "split": "text-window"}
