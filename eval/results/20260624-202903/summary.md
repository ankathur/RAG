# Retrieval mode comparison

- Generated: 2026-06-24T20:29:03
- Questions: 5  ·  k: 6  ·  modes: vector, pageindex, hybrid
- Generator: `Qwen/Qwen3.5-27B`  ·  Judge: `google/gemma-4-31B-it`  ·  Embeddings: `BAAI/bge-m3`
- Endpoint: `http://172.16.17.82/v1`

## Summary (mean per mode)

| metric | vector | pageindex | hybrid |
|---|---|---|---|
| page_hit | 0.800 | 0.800 | 0.600 |
| page_recall | 0.800 | 0.800 | 0.600 |
| page_mrr | 0.700 | 0.667 | 0.600 |
| page_ndcg | 0.726 | 0.700 | 0.600 |
| doc_hit | 0.800 | 0.800 | 0.600 |
| doc_mrr | 0.800 | 0.667 | 0.600 |
| context_precision | 0.867 | 0.250 | 1.000 |
| context_recall | 1.000 | 0.400 | 1.000 |
| faithfulness | 1.000 | 0.850 | 1.000 |
| answer_relevancy | 0.714 | 0.197 | 0.703 |
| factual_correctness | 0.248 | 0.412 | 0.160 |
| latency_s | 43.879 | 74.006 | 183.207 |

**Recommended mode (by page nDCG, tie-break context precision): `vector`**

Deterministic IR metrics (`page_*`, `doc_*`) are the hard ranking; Ragas
LLM-judge scores are directional. See `per_question.csv` for drill-down.
