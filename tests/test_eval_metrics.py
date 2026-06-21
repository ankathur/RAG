"""Offline unit tests for the deterministic IR metrics (no Ragas/LLM needed)."""

from __future__ import annotations

import math

from eval.metrics import (
    compute_ir,
    hit_at_k,
    mrr,
    page_ndcg_at_k,
    page_recall_at_k,
    parse_locator,
)


def test_parse_locator_single_page():
    assert parse_locator("p.12") == {12}
    assert parse_locator("p.2 §Working Hours") == {2}


def test_parse_locator_range_and_dashes():
    assert parse_locator("p.2-5 §2.3 Title") == {2, 3, 4, 5}
    assert parse_locator("pp.10–11") == {10, 11}  # en dash


def test_parse_locator_pageless_and_empty():
    assert parse_locator("§2.3 Risk Factors") == set()
    assert parse_locator("") == set()
    assert parse_locator(None) == set()


def test_hit_and_mrr():
    flags = [False, True, False]
    assert hit_at_k(flags, 3) == 1.0
    assert hit_at_k(flags, 1) == 0.0
    assert mrr(flags) == 0.5
    assert mrr([False, False]) == 0.0


def test_page_recall_coverage():
    retrieved = [("d1", {2}), ("d1", {3}), ("d2", {9})]
    gold_pages = {2, 3, 4}
    # 2 of 3 gold pages covered within top-3
    assert page_recall_at_k(retrieved, "d1", gold_pages, 3) == 2 / 3
    # wrong doc → no coverage
    assert page_recall_at_k(retrieved, "dX", gold_pages, 3) == 0.0
    # no gold pages → undefined
    assert math.isnan(page_recall_at_k(retrieved, "d1", set(), 3))


def test_page_ndcg_bounds_and_perfect():
    gold = {1, 2}
    perfect = [("d1", {1}), ("d1", {2}), ("d1", {9})]
    assert page_ndcg_at_k(perfect, "d1", gold, 3) == 1.0
    # relevant page only at rank 3 → less than perfect but > 0
    delayed = [("d2", {5}), ("d2", {6}), ("d1", {1})]
    score = page_ndcg_at_k(delayed, "d1", gold, 3)
    assert 0.0 < score < 1.0
    # never exceeds 1 even with redundant coverage
    redundant = [("d1", {1}), ("d1", {1}), ("d1", {2})]
    assert page_ndcg_at_k(redundant, "d1", gold, 3) <= 1.0


def test_compute_ir_keys_and_doc_level():
    retrieved = [("d2", set()), ("d1", {3})]  # d1 locator had no page
    out = compute_ir(retrieved, "d1", {3}, k=2)
    assert set(out) == {
        "page_hit",
        "page_recall",
        "page_mrr",
        "page_ndcg",
        "doc_hit",
        "doc_mrr",
    }
    assert out["page_hit"] == 1.0
    assert out["page_mrr"] == 0.5  # gold page first appears at rank 2
    assert out["doc_hit"] == 1.0
    assert out["doc_mrr"] == 0.5
