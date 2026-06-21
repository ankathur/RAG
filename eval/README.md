# Evaluation harness — vector vs pageindex vs hybrid

Measures **which retrieval mode is best** for the corpus, combining two layers:

1. **Deterministic IR metrics** (no LLM judge) — `hit / recall / MRR / nDCG`
   computed from the page locators the retrievers already return. Objective,
   free, and the *hard* ranking. (`eval/metrics.py`, unit-tested offline.)
2. **Ragas** (LLM judge) — context precision/recall + faithfulness, answer
   relevancy, factual correctness. Directional signal on answer quality.

The judge is a **different model family** (`gemma4:26b`) than the generator
(`qwen3:30b-a3b`) to reduce self-grading bias. Everything runs against the same
OpenAI-compatible endpoint the app uses (the GB10 Ollama by default).

See [`../EVAL_PLAN.md`](../EVAL_PLAN.md) for the design rationale.

## Install

```bash
source .venv/bin/activate
pip install -r requirements-eval.txt        # ragas, langchain-openai, datasets, pandas
```

On the GB10, pull the judge alongside the existing models:

```bash
ollama pull gemma4:26b      # judge (keep qwen3:30b-a3b + bge-m3)
```

## 1. Generate a test set

`gen_testset` is **page-grounded and gentle on the LLM box**: it samples N
substantive pages spread across the corpus and makes **one sequential call per
page** (concurrency 1, retry-with-backoff, early-stop after repeated failures)
asking for a self-contained question + reference answer grounded in that page.
Gold labels are the exact `(doc_id, page)` — no fuzzy matching. **Run from the
repo root** (so `.env` and `data/` resolve).

> We deliberately do **not** use Ragas' `TestsetGenerator`: its knowledge-graph
> build fires hundreds of highly concurrent LLM/embedding calls (~80 min on a
> single local Ollama) and overheated the GB10. ~N sequential calls is far
> faster and far cooler.

```bash
python -m eval.gen_testset --n 12                  # → eval/testset.jsonl
# pass the SAME files you ingested so gold labels match the index:
python -m eval.gen_testset --n 12 --corpus data/kb/tb/fda-isoniazid-label.pdf ...
python -m eval.gen_testset --n 12 --pace 1.0       # sleep 1s between calls (extra thermal headroom)
```

Useful flags: `--min-chars` (skip boilerplate/short pages, default 400),
`--max-chars` (page text sent to the model, default 4000), `--pace` (seconds
between calls), `--seed` (page-selection shuffle). Each output line is
`{question, reference_answer, reference_contexts, gold_doc_id, gold_pages,
synthesizer}`; feel free to hand-trim the resulting jsonl.

## 2. Run the comparison

```bash
python -m eval.run_eval --n 5                      # smoke: 3 modes × 5 questions
python -m eval.run_eval                            # full test set
python -m eval.run_eval --modes vector hybrid --no-ragas   # IR only, faster
```

Outputs land in `eval/results/<timestamp>/`:

- `summary.md` — modes × metrics table + recommended mode
- `summary.csv` — same, machine-readable
- `per_question.csv` — per-question drill-down (citations, latency, answer, errors)

`k` is fixed across modes (defaults to `RAG_TOP_K`) for a fair comparison. A
full run is compute-heavy (`n × 3 modes × several metrics`); start small and use
a background job for large `n`.

## Configuration (env, prefix `RAG_EVAL_`)

| Var | Default | Meaning |
|---|---|---|
| `RAG_EVAL_JUDGE_MODEL` | `gemma4:26b` | LLM judge model |
| `RAG_EVAL_JUDGE_BASE_URL` | generation endpoint | judge `/v1` base URL |
| `RAG_EVAL_JUDGE_API_KEY` | generation key | judge API key |
| `RAG_EVAL_EMBEDDING_MODEL` | `RAG_EMBEDDING_MODEL` | embeddings for Ragas |
| `RAG_EVAL_N` | `25` | questions to synthesize / evaluate |
| `RAG_EVAL_K` | `RAG_TOP_K` | retrieval depth |
| `RAG_EVAL_CORPUS_DIR` | `data/kb/tb` | corpus for test-set synthesis |
| `RAG_EVAL_TESTSET_PATH` | `eval/testset.jsonl` | test-set location |
| `RAG_EVAL_OUTPUT_DIR` | `eval/results` | results root |
| `RAG_EVAL_RAGAS_MAX_WORKERS` | `1` | Ragas eval concurrency (1 = sequential, gentlest on a local box) |

The pipeline under test (generator, retrievers, embeddings, `k`) is configured
through the usual `RAG_*` vars / `.env`.

## Metrics

| Metric | Layer | Notes |
|---|---|---|
| `page_hit`, `page_recall`, `page_mrr`, `page_ndcg` | IR | page-level (locator covers a gold page) |
| `doc_hit`, `doc_mrr` | IR | doc-level (right document, any page) |
| `context_precision`, `context_recall` | Ragas | retrieval quality |
| `faithfulness`, `answer_relevancy`, `factual_correctness` | Ragas | answer quality |
| `latency_s` | — | wall-clock per `ask` |

Offline metric tests: `pytest tests/test_eval_metrics.py`.
