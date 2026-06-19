"""PageIndex: vectorless, reasoning-based retrieval (see SPEC.md §9.2).

Two steps, both driven by the reasoning LLM:
  1. Build  — turn each document into a table-of-contents *tree* of sections.
  2. Search — have the LLM reason over the tree(s) and pick relevant nodes,
              like a human navigating a book's contents.

Trees (and the page text they point at) are persisted as self-contained JSON so
retrieval works across restarts without the original files.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.config import Settings, get_settings
from rag.llm.base import LLMProvider
from rag.llm.factory import build_llm
from rag.models import Document, RetrievedContext, TreeNode

_MAX_CONTEXT_CHARS = 4000

_BUILD_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "start_page": {"type": "integer"},
                    "end_page": {"type": "integer"},
                    "level": {"type": "integer"},
                },
                "required": ["title", "start_page", "end_page"],
            },
        }
    },
    "required": ["sections"],
}

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_node_ids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["selected_node_ids"],
}


class PageIndexRetriever:
    name = "pageindex"

    def __init__(
        self, settings: Settings | None = None, llm: LLMProvider | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self._docs: dict[str, Document] = {}
        self._trees: dict[str, list[TreeNode]] = {}

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = build_llm("reasoning", self.settings)
        return self._llm

    # -- Retriever --------------------------------------------------------------
    def index(self, docs: list[Document]) -> None:
        Path(self.settings.index_dir).mkdir(parents=True, exist_ok=True)
        for doc in docs:
            roots = self._build_tree(doc)
            self._docs[doc.id] = doc
            self._trees[doc.id] = roots
            self._persist_doc(doc, roots)

    def select_sections(self, query: str, k: int) -> list[dict]:
        """Reason over the tree(s) and return ordered selected node infos.

        Each dict has: doc_id, title, summary, start_page, end_page. Used by both
        ``retrieve`` and the hybrid ``pageindex_then_vector`` strategy.
        """
        if not self._trees:
            self.load()
        if not self._trees:
            return []

        serialized, node_map = self._serialize_all()
        prompt = (
            "You retrieve information from documents by navigating their "
            "table-of-contents trees, the way a human uses a book's contents page. "
            "Each node is shown as `id · pages · title — summary`.\n\n"
            f"Question: {query}\n\n"
            f"Document trees:\n{serialized}\n\n"
            f"Select the node ids whose sections most likely contain the answer. "
            f"Prefer specific leaf sections over broad parents. Return up to {k * 2} "
            "ids ordered by relevance, plus a short rationale."
        )
        try:
            data = self.llm.structured(
                prompt, _SEARCH_SCHEMA, schema_name="node_selection"
            )
            selected = [str(x) for x in data.get("selected_node_ids", [])]
        except Exception:
            selected = []

        if not selected:
            selected = self._lexical_fallback(query, node_map, k)

        infos: list[dict] = []
        seen: set[str] = set()
        for gid in selected:
            if gid in seen or gid not in node_map:
                continue
            seen.add(gid)
            infos.append(node_map[gid])
        return infos

    def retrieve(self, query: str, k: int) -> list[RetrievedContext]:
        out: list[RetrievedContext] = []
        for info in self.select_sections(query, k):
            doc = self._docs.get(info["doc_id"])
            if doc is None:
                continue
            text = doc.text_for_range(info["start_page"], info["end_page"])
            if not text.strip():
                continue
            out.append(
                RetrievedContext(
                    text=text[:_MAX_CONTEXT_CHARS],
                    doc_id=info["doc_id"],
                    locator=_locator(info),
                    score=None,
                    origin="pageindex",
                )
            )
            if len(out) >= k:
                break
        return out

    def load(self) -> None:
        index_dir = Path(self.settings.index_dir)
        if not index_dir.exists():
            return
        for path in sorted(index_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                doc = Document.model_validate(payload["doc"])
                roots = [TreeNode.model_validate(n) for n in payload["tree"]]
            except Exception:
                continue
            self._docs[doc.id] = doc
            self._trees[doc.id] = roots

    # -- build ------------------------------------------------------------------
    def _build_tree(self, doc: Document) -> list[TreeNode]:
        page_count = len(doc.pages) or 1
        digest = _page_digest(doc)
        outline_hint = ""
        if doc.metadata.get("outline"):
            entries = "; ".join(
                f"{o['title']} (p.{o['page']})" for o in doc.metadata["outline"][:50]
            )
            outline_hint = f"\nExisting bookmarks you may use as structure: {entries}\n"

        prompt = (
            "Build a table-of-contents tree index for the document below so it can "
            "be navigated like a book. Produce contiguous, non-overlapping sections "
            f"covering pages 1..{page_count}. For each: a title, a 1-2 sentence "
            "summary of what it contains, start_page and end_page (1-based, within "
            f"1..{page_count}), and a nesting level (1 = top-level). Aim for at most "
            f"{self.settings.pageindex_max_pages_per_node} pages per leaf section."
            f"{outline_hint}\n\nPage digest:\n{digest}"
        )
        try:
            data = self.llm.structured(prompt, _BUILD_SCHEMA, schema_name="toc_tree")
            sections = data.get("sections", [])
        except Exception:
            sections = []

        roots = _tree_from_sections(sections, page_count)
        if not roots:
            roots = _fallback_tree(page_count, self.settings.pageindex_max_pages_per_node)
        return roots

    # -- persistence ------------------------------------------------------------
    def _persist_doc(self, doc: Document, roots: list[TreeNode]) -> None:
        path = Path(self.settings.index_dir) / f"{doc.id}.json"
        payload = {
            "doc": doc.model_dump(),
            "tree": [n.model_dump() for n in roots],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # -- serialization ----------------------------------------------------------
    def _serialize_all(self) -> tuple[str, dict[str, dict]]:
        lines: list[str] = []
        node_map: dict[str, dict] = {}
        for doc_id, roots in self._trees.items():
            fname = self._docs[doc_id].metadata.get("filename", doc_id)
            lines.append(f"# Document: {fname} (id={doc_id})")
            for root in roots:
                _serialize_node(root, doc_id, 0, lines, node_map)
        return "\n".join(lines), node_map

    def _lexical_fallback(
        self, query: str, node_map: dict[str, dict], k: int
    ) -> list[str]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored: list[tuple[int, str]] = []
        for gid, info in node_map.items():
            hay = f"{info['title']} {info['summary']}".lower()
            score = sum(1 for t in terms if t in hay)
            if score:
                scored.append((score, gid))
        scored.sort(reverse=True)
        return [gid for _, gid in scored[:k]]


# --- module helpers ------------------------------------------------------------
def _locator(info: dict) -> str:
    if info["start_page"] == info["end_page"]:
        pages = f"p.{info['start_page']}"
    else:
        pages = f"p.{info['start_page']}-{info['end_page']}"
    return f"{pages} §{info['title']}"


def _page_digest(doc: Document) -> str:
    pages = doc.pages or []
    budget = max(80, 12000 // max(1, len(pages)))
    lines = []
    for page in pages:
        snippet = " ".join(page.text[:budget].split())
        lines.append(f"[p.{page.number}] {snippet}")
    return "\n".join(lines)


def _serialize_node(
    node: TreeNode, doc_id: str, depth: int, lines: list[str], node_map: dict[str, dict]
) -> None:
    gid = f"{doc_id}#{node.node_id}"
    indent = "  " * depth
    pages = (
        f"p.{node.start_page}"
        if node.start_page == node.end_page
        else f"p.{node.start_page}-{node.end_page}"
    )
    summary = node.summary.strip().replace("\n", " ")
    lines.append(f"{indent}- {gid} · {pages} · {node.title} — {summary}")
    node_map[gid] = {
        "doc_id": doc_id,
        "title": node.title,
        "summary": node.summary,
        "start_page": node.start_page,
        "end_page": node.end_page,
    }
    for child in node.nodes:
        _serialize_node(child, doc_id, depth + 1, lines, node_map)


def _tree_from_sections(sections: list[dict], page_count: int) -> list[TreeNode]:
    """Convert a flat list of sections (with levels) into a nested tree."""
    cleaned: list[dict] = []
    for s in sections:
        try:
            start = max(1, min(int(s["start_page"]), page_count))
            end = max(start, min(int(s["end_page"]), page_count))
        except (KeyError, TypeError, ValueError):
            continue
        title = str(s.get("title", "")).strip() or f"Pages {start}-{end}"
        cleaned.append(
            {
                "title": title,
                "summary": str(s.get("summary", "")).strip(),
                "start_page": start,
                "end_page": end,
                "level": int(s.get("level", 1) or 1),
            }
        )
    if not cleaned:
        return []

    cleaned.sort(key=lambda s: (s["start_page"], s["level"]))
    counter = {"n": 0}

    def _new_node(s: dict) -> TreeNode:
        counter["n"] += 1
        return TreeNode(
            node_id=f"{counter['n']:04d}",
            title=s["title"],
            summary=s["summary"],
            start_page=s["start_page"],
            end_page=s["end_page"],
        )

    roots: list[TreeNode] = []
    stack: list[tuple[int, TreeNode]] = []  # (level, node)
    for s in cleaned:
        node = _new_node(s)
        while stack and stack[-1][0] >= s["level"]:
            stack.pop()
        if stack:
            stack[-1][1].nodes.append(node)
        else:
            roots.append(node)
        stack.append((s["level"], node))
    return roots


def _fallback_tree(page_count: int, max_pages: int) -> list[TreeNode]:
    """Deterministic tree: contiguous page groups when the LLM yields nothing."""
    roots: list[TreeNode] = []
    n = 0
    start = 1
    while start <= page_count:
        end = min(start + max_pages - 1, page_count)
        n += 1
        roots.append(
            TreeNode(
                node_id=f"{n:04d}",
                title=f"Pages {start}-{end}",
                summary="",
                start_page=start,
                end_page=end,
            )
        )
        start = end + 1
    return roots
