# Retrieval mode comparison

- Generated: 2026-06-25T20:50:38
- Questions: 12  ·  k: 6  ·  modes: vector, pageindex, hybrid
- Generator: `google/gemma-4-31B-it`  ·  Judge: `google/gemma-4-31B-it`  ·  Embeddings: `BAAI/bge-m3`
- Endpoint: `http://172.16.17.82/v1`

## Summary (mean per mode)

| metric | vector | pageindex | hybrid |
|---|---|---|---|
| page_hit | 0.833 | 0.833 | 0.833 |
| page_recall | 0.833 | 0.833 | 0.833 |
| page_mrr | 0.521 | 0.694 | 0.625 |
| page_ndcg | 0.599 | 0.730 | 0.680 |
| doc_hit | 0.917 | 0.833 | 1.000 |
| doc_mrr | 0.833 | 0.833 | 0.875 |
| context_precision | 0.713 | 0.486 | 0.692 |
| context_recall | 0.917 | 0.583 | 0.750 |
| faithfulness | 1.000 | 1.000 | 0.979 |
| answer_relevancy | 0.928 | 0.551 | 0.861 |
| factual_correctness | 0.253 | 0.460 | 0.208 |
| latency_s | 1.076 | 2.842 | 2.994 |

**Recommended mode (by page nDCG, tie-break context precision): `pageindex`**

Deterministic IR metrics (`page_*`, `doc_*`) are the hard ranking; Ragas
LLM-judge scores are directional. See `per_question.csv` for drill-down.
