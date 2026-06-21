# Evaluation Plan: vector vs pageindex vs hybrid retrieval

## Context
We want to measure **which retrieval mode is best** for the corpus, not just eyeball it. The
approach combines two layers:

1. **Objective retrieval metrics** (no LLM judge) — hit@k / recall@k / MRR / nDCG@k computed from
   the `locator`s our retrievers already return (`RetrievedContext.doc_id` + page). This gives an
   unbiased ranking of vector vs pageindex vs hybrid.
2. **Ragas** for LLM-judge RAG quality — context precision/recall (retrieval) and faithfulness /
   answer relevancy / factual correctness (answers). Ragas is purpose-built for RAG, lightweight,
   and provider-neutral (runs against any OpenAI-compatible endpoint, i.e. the GB10 Ollama).

Chosen knobs:
- **Judge = `gemma4:26b` on the GB10** — a *different* model family than the qwen3 generator, which
  reduces self-grading bias.
- **Test set = LLM-synthesized** from the corpus via Ragas, with gold doc/page labels recovered
  from each question's reference context (so the objective IR metrics still work).

No application code changes — this is an additive `eval/` package that imports the existing
pipeline. Reuses: `rag.pipeline.RAGPipeline.ask` (returns `Answer.text` + `.citations`),
`rag.factory.build_retriever`, `rag.documents.loader.load_documents` (pages for gold mapping),
`rag.config.Settings`, and `rag.models.RetrievedContext` (`.locator`, `.doc_id`).

## Why these tools
- **Ragas** — RAG-specific metric set, dual-judge design (resists prompt sensitivity), works with
  Ollama via OpenAI-compatible base_url / LiteLLM adapter. Ideal for fast retriever comparison.
- **Deterministic IR harness** — objective, free, no judge bias; directly answers "which mode
  retrieves the right pages." (Research caveat: LLM judges can misrank hard-negative contexts, so
  the deterministic layer is the tiebreaker.)
- **Separate judge model (gemma4:26b)** — decorrelates generator and grader.
- DeepEval (pytest/CI) and promptfoo (YAML/UI) were considered; Ragas + IR best fits a one-off
  comparison with both objective and quality signals. DeepEval can be layered later for CI gating.

## Components to build (new `eval/` package)
- `eval/settings.py` — eval config from env (with defaults): judge model/base_url
  (`gemma4:26b` @ GB10), embeddings (`bge-m3` @ GB10), `k` (= `Settings.top_k`), `n` questions,
  corpus dir (`data/kb/tb`), output dir. Reuses `rag.config.Settings` for the pipeline side.
- `eval/gen_testset.py` — **page-grounded** generation (implemented): sample N substantive
  pages spread across the corpus and make **one sequential LLM call per page** (reusing
  `rag.llm`'s `structured()`) for a self-contained `{question, reference_answer}` grounded in
  that page → `eval/testset.jsonl`. **Gold labels are the exact `(doc_id, page)`** of the
  source page — no fuzzy matching. Light validity filter drops weak/source-leaking questions.
  > Note: the original plan used Ragas' `TestsetGenerator`, but its knowledge-graph build fires
  > hundreds of highly concurrent LLM/embedding calls (~80 min on one local Ollama) and
  > **overheated the GB10**. The sequential page-grounded generator (≈N calls, concurrency 1,
  > retry-with-backoff, early-stop on repeated failures) is far faster and far cooler. Ragas is
  > still used for the *evaluation* metrics in `run_eval` (with `max_workers=1`).
- `eval/metrics.py` — `parse_locator()` ("p.N", "p.A-B §title" → page set) + IR metrics
  `hit_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k` (page-level and doc-level).
- `eval/run_eval.py` — for each `mode in {vector, pageindex, hybrid}` and each question:
  call `RAGPipeline.ask(q, mode=mode)`, capture `answer.text`, `answer.citations`, and latency;
  compute deterministic IR from citations vs gold; compute Ragas metrics (judge=gemma4,
  emb=bge-m3). Write `eval/results/<timestamp>/`: `per_question.csv`, `summary.csv`, and
  `summary.md` (modes × metrics comparison table) ranking the modes. `--n` and `--modes` flags.
- `eval/README.md`, `requirements-eval.txt` (`ragas`, `langchain-openai`, `datasets`, `pandas`),
  and a `[project.optional-dependencies] eval = [...]` entry in `pyproject.toml` (keep core light).

## Metrics reported per mode
- Objective retrieval: **Hit@k, Recall@k, MRR, nDCG@k** (page-level + doc-level).
- Ragas retrieval: **context_precision, context_recall**.
- Ragas answer: **faithfulness, answer_relevancy, factual_correctness** (vs reference answer).
- **Latency** (s/query). Output = one comparison table + per-question drill-down.

## Setup / prereqs
- On GB10: `ollama pull gemma4:26b` (judge); keep `qwen3:30b-a3b` + `bge-m3`
  (~40 GB total, fits 128 GB).
- `pip install -r requirements-eval.txt` in `.venv`.
- Consider raising Ollama `num_ctx` for judge/gen on long contexts.

## Caveats
- Synthetic questions need filtering — include a light validity check; optional manual trim of
  `testset.jsonl`.
- Local LLM-judge scores are **directional**; treat the deterministic IR metrics as the hard ranking.
- Full run is GB10-compute-heavy (n × 3 modes × several metrics × dual-judge). Start `n≈20–30`,
  run as a background job, and fix `k` across modes for a fair comparison.

## Verification
1. `ollama list` on GB10 shows `gemma4:26b`, `qwen3:30b-a3b`, `bge-m3`.
2. `python -m eval.gen_testset --n 25` → `eval/testset.jsonl` with ≥20 rows, each carrying gold
   doc/page.
3. `python -m eval.run_eval --n 5` (smoke) → `eval/results/<ts>/summary.md` populated for all 3
   modes; all metric values within [0,1]; latencies sane.
4. Full run → inspect `summary.md` ranking + `per_question.csv` for failure cases.
5. `pytest -q` still green (eval is additive; offline unit tests unaffected).

## Sources
- Ragas — context precision/recall, LLM adapters/Ollama:
  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/ ,
  https://docs.ragas.io/en/stable/howtos/llm-adapters/
- Framework comparison (Ragas/TruLens/DeepEval/promptfoo) + judge caveat:
  https://atlan.com/know/llm-evaluation-frameworks-compared/ ,
  https://deepeval.com/blog/deepeval-vs-ragas
- DeepEval RAG metrics / Ollama (for the optional CI layer):
  https://deepeval.com/docs/getting-started-rag , https://deepeval.com/integrations/models/ollama
