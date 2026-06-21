"""Deterministic information-retrieval metrics (no LLM judge).

Pure-stdlib so the objective layer can be unit-tested offline, with no Ragas
install and no live endpoint. These metrics are the *hard* ranking of the three
modes; the Ragas LLM-judge scores in :mod:`eval.run_eval` are directional.

A *retrieved* list is an ordered ``[(doc_id, pages), ...]`` (rank 0 first),
where ``pages`` is the ``set[int]`` parsed from a citation's locator. *Gold* is
a single ``gold_doc`` plus the ``gold_pages`` set recovered for a question.

Two granularities:

* **page-level** — relevant = same doc *and* the locator covers a gold page.
* **doc-level** — relevant = same doc (locators without a page still count).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

__all__ = [
    "parse_locator",
    "hit_at_k",
    "mrr",
    "page_recall_at_k",
    "page_ndcg_at_k",
    "compute_ir",
    "IR_METRICS",
]

# IR metric column names produced by ``compute_ir`` (k is a run-level constant).
IR_METRICS = [
    "page_hit",
    "page_recall",
    "page_mrr",
    "page_ndcg",
    "doc_hit",
    "doc_mrr",
]

# Matches "p.12", "pp. 3", "p.2-5", "p.2–5" (any dash). Page words elsewhere in
# a section title (e.g. "§page layout") won't match because the "p." anchor is
# required.
_PAGE_RE = re.compile(r"pp?\.\s*(\d+)\s*(?:[-‐-―]\s*(\d+))?", re.IGNORECASE)
_MAX_RANGE = 5000  # guard against a malformed locator exploding into a huge set

Retrieved = Sequence[tuple[str, set[int]]]


def parse_locator(locator: str | None) -> set[int]:
    """Extract the page numbers referenced by a citation locator.

    Handles ``"p.12"``, ``"p.2-5 §2.3 Title"``, ``"pp. 3, 7"`` (each
    ``p.``-anchored number), and returns an empty set for page-less locators
    such as ``"§2.3 Risk Factors"``.
    """
    pages: set[int] = set()
    if not locator:
        return pages
    for m in _PAGE_RE.finditer(locator):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if end < start:
            start, end = end, start
        if end - start <= _MAX_RANGE:
            pages.update(range(start, end + 1))
    return pages


def _page_rel_flags(
    retrieved: Retrieved, gold_doc: str | None, gold_pages: set[int]
) -> list[bool]:
    return [d == gold_doc and bool(p & gold_pages) for d, p in retrieved]


def _doc_rel_flags(retrieved: Retrieved, gold_doc: str | None) -> list[bool]:
    return [d == gold_doc for d, _ in retrieved]


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    """1.0 if any of the first ``k`` items is relevant, else 0.0."""
    return 1.0 if any(flags[:k]) else 0.0


def mrr(flags: Sequence[bool]) -> float:
    """Reciprocal rank of the first relevant item (0.0 if none)."""
    for i, f in enumerate(flags, start=1):
        if f:
            return 1.0 / i
    return 0.0


def page_recall_at_k(
    retrieved: Retrieved, gold_doc: str | None, gold_pages: set[int], k: int
) -> float:
    """Fraction of gold pages covered by the top-``k`` retrieved locators.

    ``nan`` when there are no gold pages (page-level recall is undefined).
    """
    if not gold_pages:
        return float("nan")
    covered: set[int] = set()
    for d, p in retrieved[:k]:
        if d == gold_doc:
            covered |= p & gold_pages
    return len(covered) / len(gold_pages)


def page_ndcg_at_k(
    retrieved: Retrieved, gold_doc: str | None, gold_pages: set[int], k: int
) -> float:
    """nDCG@k with binary *new-coverage* gains.

    An item scores 1 only when it covers a gold page no higher-ranked item
    already covered, so the number of gainful positions never exceeds the gold
    page count and nDCG stays in ``[0, 1]``. ``nan`` when no gold pages.
    """
    if not gold_pages:
        return float("nan")
    covered: set[int] = set()
    dcg = 0.0
    for i, (d, p) in enumerate(retrieved[:k], start=1):
        if d == gold_doc:
            new = (p & gold_pages) - covered
            if new:
                covered |= new
                dcg += 1.0 / math.log2(i + 1)
    ideal = min(len(gold_pages), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else float("nan")


def compute_ir(
    retrieved: Retrieved, gold_doc: str | None, gold_pages: set[int], k: int
) -> dict[str, float]:
    """All deterministic IR metrics for one question, as a flat dict."""
    pflags = _page_rel_flags(retrieved, gold_doc, gold_pages)
    dflags = _doc_rel_flags(retrieved, gold_doc)
    return {
        "page_hit": hit_at_k(pflags, k),
        "page_recall": page_recall_at_k(retrieved, gold_doc, gold_pages, k),
        "page_mrr": mrr(pflags[:k]),
        "page_ndcg": page_ndcg_at_k(retrieved, gold_doc, gold_pages, k),
        "doc_hit": hit_at_k(dflags, k),
        "doc_mrr": mrr(dflags[:k]),
    }
