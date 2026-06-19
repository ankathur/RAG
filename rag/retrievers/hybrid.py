"""Hybrid retrieval: combine vector + PageIndex (see SPEC.md §9.3)."""

from __future__ import annotations

from rag.config import Settings, get_settings
from rag.llm.base import LLMProvider
from rag.llm.factory import build_llm
from rag.models import Document, RetrievedContext
from rag.retrievers.base import Retriever
from rag.retrievers.pageindex import PageIndexRetriever
from rag.retrievers.vector import VectorRetriever

_RERANK_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "string"}}},
    "required": ["ranking"],
}


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(
        self,
        settings: Settings | None = None,
        vector: VectorRetriever | None = None,
        pageindex: PageIndexRetriever | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector = vector or VectorRetriever(self.settings)
        self.pageindex = pageindex or PageIndexRetriever(self.settings)
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = build_llm("reasoning", self.settings)
        return self._llm

    # -- Retriever --------------------------------------------------------------
    def index(self, docs: list[Document]) -> None:
        self.vector.index(docs)
        self.pageindex.index(docs)

    def load(self) -> None:
        self.vector.load()
        self.pageindex.load()

    def retrieve(self, query: str, k: int) -> list[RetrievedContext]:
        if self.settings.hybrid_strategy == "pageindex_then_vector":
            return self._pageindex_then_vector(query, k)
        return self._merge_rerank(query, k)

    # -- strategies -------------------------------------------------------------
    def _merge_rerank(self, query: str, k: int) -> list[RetrievedContext]:
        vec = self.vector.retrieve(query, k)
        pi = self.pageindex.retrieve(query, k)

        # Dedupe by (doc_id, locator), keeping first occurrence.
        merged: list[RetrievedContext] = []
        seen: set[tuple[str, str]] = set()
        for ctx in [*vec, *pi]:
            key = ctx.dedup_key()
            if key in seen:
                continue
            seen.add(key)
            merged.append(ctx)

        if len(merged) <= k:
            return merged

        ordered = self._llm_rerank(query, merged, k)
        return ordered or self._interleave(vec, pi, k)

    def _pageindex_then_vector(self, query: str, k: int) -> list[RetrievedContext]:
        sections = self.pageindex.select_sections(query, k)
        if not sections:
            return self.vector.retrieve(query, k)

        doc_ids = sorted({s["doc_id"] for s in sections})
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        candidates = self.vector.retrieve(query, k * 4, where=where)

        allowed = _allowed_pages(sections)
        filtered = [c for c in candidates if _page_in(c, allowed)]
        result = filtered[:k]

        if len(result) < k:  # top up with the PageIndex sections themselves
            section_ctxs = self.pageindex.retrieve(query, k)
            seen = {c.dedup_key() for c in result}
            for ctx in section_ctxs:
                if ctx.dedup_key() not in seen:
                    result.append(ctx)
                if len(result) >= k:
                    break
        return result[:k]

    # -- helpers ----------------------------------------------------------------
    def _llm_rerank(
        self, query: str, candidates: list[RetrievedContext], k: int
    ) -> list[RetrievedContext]:
        labelled = {f"c{i}": c for i, c in enumerate(candidates)}
        listing = "\n".join(
            f"{cid} [{c.origin} {c.locator}] {c.text[:300].strip()}"
            for cid, c in labelled.items()
        )
        prompt = (
            "Rank the candidate passages by how well they answer the question. "
            "Return the ids ordered best-first; drop irrelevant ones.\n\n"
            f"Question: {query}\n\nCandidates:\n{listing}"
        )
        try:
            data = self.llm.structured(prompt, _RERANK_SCHEMA, schema_name="rerank")
            order = [str(x) for x in data.get("ranking", [])]
        except Exception:
            return []
        ranked: list[RetrievedContext] = []
        for cid in order:
            if cid in labelled and labelled[cid] not in ranked:
                ranked.append(labelled[cid])
            if len(ranked) >= k:
                break
        return ranked

    @staticmethod
    def _interleave(
        a: list[RetrievedContext], b: list[RetrievedContext], k: int
    ) -> list[RetrievedContext]:
        out: list[RetrievedContext] = []
        seen: set[tuple[str, str]] = set()
        for pair in zip(a, b):
            for ctx in pair:
                key = ctx.dedup_key()
                if key not in seen:
                    seen.add(key)
                    out.append(ctx)
        for ctx in [*a, *b]:  # drain leftovers
            key = ctx.dedup_key()
            if key not in seen:
                seen.add(key)
                out.append(ctx)
        return out[:k]


def _allowed_pages(sections: list[dict]) -> dict[str, set[int]]:
    allowed: dict[str, set[int]] = {}
    for s in sections:
        pages = allowed.setdefault(s["doc_id"], set())
        pages.update(range(s["start_page"], s["end_page"] + 1))
    return allowed


def _page_in(ctx: RetrievedContext, allowed: dict[str, set[int]]) -> bool:
    pages = allowed.get(ctx.doc_id)
    if not pages:
        return False
    # Vector locators are "p.<n>"; parse the page number.
    loc = ctx.locator
    if loc.startswith("p.") and loc[2:].isdigit():
        return int(loc[2:]) in pages
    return True
