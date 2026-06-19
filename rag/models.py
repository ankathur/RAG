"""Core data models shared across the system (see SPEC.md §5)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocFormat = Literal["pdf", "md", "txt"]
Origin = Literal["vector", "pageindex"]


class Page(BaseModel):
    number: int  # 1-based page (or pseudo-page) number
    text: str


class Document(BaseModel):
    id: str
    source_path: str
    format: DocFormat
    pages: list[Page] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def text_for_range(self, start_page: int, end_page: int) -> str:
        """Return concatenated text for an inclusive 1-based page range."""
        lo, hi = min(start_page, end_page), max(start_page, end_page)
        return "\n\n".join(p.text for p in self.pages if lo <= p.number <= hi)


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    page: int | None = None
    position: int = 0


class TreeNode(BaseModel):
    """A node in a PageIndex table-of-contents tree."""

    node_id: str
    title: str
    summary: str = ""
    start_page: int = 1
    end_page: int = 1
    nodes: list["TreeNode"] = Field(default_factory=list)

    def walk(self):
        """Yield this node and all descendants, depth-first."""
        yield self
        for child in self.nodes:
            yield from child.walk()


class RetrievedContext(BaseModel):
    text: str
    doc_id: str
    locator: str  # e.g. "p.12" or "§2.3 Risk Factors"
    score: float | None = None
    origin: Origin = "vector"

    def dedup_key(self) -> tuple[str, str]:
        return (self.doc_id, self.locator)


class Answer(BaseModel):
    text: str
    citations: list[RetrievedContext] = Field(default_factory=list)
    mode: str = "hybrid"
    usage: dict[str, Any] = Field(default_factory=dict)
