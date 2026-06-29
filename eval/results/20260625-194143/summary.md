# Retrieval mode comparison

- Generated: 2026-06-25T19:41:43
- Questions: 5  ·  k: 6  ·  modes: vector, pageindex, hybrid
- Generator: `google/gemma-4-31B-it`  ·  Judge: `Qwen/Qwen3.5-27B`  ·  Embeddings: `BAAI/bge-m3`
- Endpoint: `http://172.16.17.43/v1`

## Summary (mean per mode)

| metric | vector | pageindex | hybrid |
|---|---|---|---|
| page_hit | 0.600 | 0.800 | 0.600 |
| page_recall | 0.600 | 0.800 | 0.600 |
| page_mrr | 0.350 | 0.700 | 0.400 |
| page_ndcg | 0.412 | 0.726 | 0.452 |
| doc_hit | 0.800 | 0.800 | 1.000 |
| doc_mrr | 0.700 | 0.800 | 0.800 |
| context_precision | — | — | — |
| context_recall | — | — | — |
| faithfulness | — | — | — |
| answer_relevancy | — | — | — |
| factual_correctness | — | — | — |
| latency_s | 1.189 | 2.261 | 2.653 |

**Recommended mode (by page nDCG, tie-break context precision): `pageindex`**

Deterministic IR metrics (`page_*`, `doc_*`) are the hard ranking; Ragas
LLM-judge scores are directional. See `per_question.csv` for drill-down.
