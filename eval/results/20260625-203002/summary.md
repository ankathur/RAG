# Retrieval mode comparison

- Generated: 2026-06-25T20:30:02
- Questions: 5  ·  k: 6  ·  modes: vector, pageindex, hybrid
- Generator: `google/gemma-4-31B-it`  ·  Judge: `google/gemma-4-31B-it`  ·  Embeddings: `BAAI/bge-m3`
- Endpoint: `http://172.16.17.82/v1`

## Summary (mean per mode)

| metric | vector | pageindex | hybrid |
|---|---|---|---|
| page_hit | 0.600 | 0.800 | 0.600 |
| page_recall | 0.600 | 0.800 | 0.600 |
| page_mrr | 0.350 | 0.700 | 0.400 |
| page_ndcg | 0.412 | 0.726 | 0.452 |
| doc_hit | 0.800 | 0.800 | 1.000 |
| doc_mrr | 0.700 | 0.800 | 0.800 |
| context_precision | 0.683 | 0.400 | 0.540 |
| context_recall | 0.800 | 0.400 | 0.600 |
| faithfulness | 1.000 | 1.000 | 0.950 |
| answer_relevancy | 0.900 | 0.354 | 0.919 |
| factual_correctness | 0.234 | 0.502 | 0.300 |
| latency_s | 1.046 | 1.837 | 2.464 |

**Recommended mode (by page nDCG, tie-break context precision): `pageindex`**

Deterministic IR metrics (`page_*`, `doc_*`) are the hard ranking; Ragas
LLM-judge scores are directional. See `per_question.csv` for drill-down.
