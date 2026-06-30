# Retrieval mode comparison

- Generated: 2026-06-26T20:10:23
- Questions: 12  ·  k: 6  ·  modes: vector, pageindex, hybrid
- Generator: `google/gemma-4-31B-it`  ·  Judge: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`  ·  Embeddings: `BAAI/bge-m3`

## Summary (mean per mode)

| metric | vector | pageindex | hybrid |
|---|---|---|---|
| page_hit | 0.833 | 0.833 | 0.833 |
| page_recall | 0.833 | 0.833 | 0.833 |
| page_mrr | 0.521 | 0.694 | 0.625 |
| page_ndcg | 0.599 | 0.730 | 0.680 |
| doc_hit | 0.917 | 0.833 | 1.000 |
| doc_mrr | 0.833 | 0.833 | 0.875 |
| context_precision | 0.724 | 0.528 | 0.776 |
| context_recall | 0.917 | 0.583 | 0.750 |
| faithfulness | 0.896 | 0.757 | 0.846 |
| answer_relevancy | 0.920 | 0.627 | 0.845 |
| factual_correctness | 0.255 | 0.381 | 0.218 |
| latency_s | 1.085 | 2.865 | 3.236 |

**Recommended mode (by page nDCG, tie-break context precision): `pageindex`**

Deterministic IR metrics (`page_*`, `doc_*`) are the hard ranking; Ragas
LLM-judge scores are directional. See `per_question.csv` for drill-down.
