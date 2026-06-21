# RAG System

A domain-agnostic Retrieval-Augmented Generation service with **three interchangeable
retrieval modes** behind one REST API:

- **vector** — chunk → embed → similarity search (Chroma + local embeddings)
- **pageindex** — vectorless, reasoning-based retrieval: build a table-of-contents tree per
  document, then have the LLM reason/tree-search over it (after [PageIndex](https://github.com/VectifyAI/PageIndex))
- **hybrid** — run both, merge, LLM-rerank

The LLM is reached through **one OpenAI-compatible client**, so any model at an
OpenAI-compatible `/v1` endpoint works by setting `base_url` + `api_key` + `model` (Ollama,
LM Studio, vLLM, llama.cpp, OpenAI, OpenRouter, Together, Groq, Claude-compat, …).
**Defaults are fully local (Ollama) — no API keys, offline.**

See [`SPEC.md`](SPEC.md) for the full specification.

## Quick start

```bash
# 1. Environment (3.12 recommended; torch/chromadb wheels may lag on 3.14)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # edit if pointing at a non-local endpoint

# 2. A local OpenAI-compatible LLM (default)
#    https://ollama.com — then:
ollama serve &
ollama pull qwen2.5:7b-instruct

# 3. Run the API
uvicorn app.api:app --reload
```

## Usage

```bash
# Health
curl localhost:8000/health

# Ingest a document (file upload) — builds vector + pageindex indexes
curl -F "file=@data/sample/handbook.md" localhost:8000/ingest
# ...or by path:
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"paths": ["data/sample/handbook.md"]}'

# Ask (mode optional; defaults to RAG_RETRIEVAL_MODE)
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"query": "How much annual leave do employees get?", "mode": "hybrid"}'
```

## Web UI

A React + Vite single-page app lives in [`web/`](web/) — drag-and-drop ingest, a
mode selector (vector/pageindex/hybrid), a chat view that shows answers with
their source citations, and a **⚙ settings panel** to point the LLM/embedding
endpoints at any OpenAI-compatible server at runtime (applied live, optionally
saved to `.env`). The same config is available over the API via `GET`/`PUT /config`.

```bash
uvicorn app.api:app --reload          # backend on :8000

cd web
npm install
npm run dev                           # UI on http://localhost:5173
```

In dev the Vite proxy forwards API calls to `:8000`, so there's nothing else to
configure. See [`web/README.md`](web/README.md) for production builds. (The API
sends permissive CORS headers so a separately-hosted build can call it too.)

## Switching the model / endpoint

Everything is config (`.env`, prefix `RAG_`). Point any role at any OpenAI-compatible endpoint
without code changes — e.g. run reasoning on a stronger remote model, generation locally:

```bash
RAG_REASONING_BASE_URL=https://openrouter.ai/api/v1
RAG_REASONING_API_KEY=sk-...
RAG_REASONING_MODEL=meta-llama/llama-3.1-70b-instruct
# generation_* left at the local Ollama default
```

## Evaluation

Benchmark the three modes against each other on your corpus. The harness in
[`eval/`](eval/) combines **deterministic IR metrics** (hit / recall / MRR /
nDCG from the page locators each retriever returns — objective, no judge) with
**Ragas** LLM-judge metrics (context precision/recall, faithfulness, answer
relevancy, factual correctness). The judge is a *different* model family
(`gemma4:26b`) than the generator to reduce self-grading bias.

Test-set synthesis is **page-grounded and gentle on a local LLM box** — one
sequential call per sampled page (not Ragas' heavy, highly-concurrent
knowledge-graph build), and Ragas evaluation runs sequentially by default.

```bash
pip install -r requirements-eval.txt     # ragas + pinned langchain 0.3 stack
ollama pull gemma4:26b                    # judge (on the box serving your models)

# 1. Synthesize a test set (one sequential call per page; exact gold doc/page)
python -m eval.gen_testset --n 12         # → eval/testset.jsonl

# 2. Compare the modes (fixed k across modes for fairness)
python -m eval.run_eval --n 5             # smoke: 3 modes × 5 questions
python -m eval.run_eval                    # full test set
python -m eval.run_eval --modes vector hybrid --no-ragas   # IR only, faster
```

Results land in `eval/results/<timestamp>/`: `summary.md` (modes × metrics table
+ recommended mode), `summary.csv`, and `per_question.csv`. Run from the repo
root so `.env`/`data/` resolve. See [`eval/README.md`](eval/README.md) for all
options and [`EVAL_PLAN.md`](EVAL_PLAN.md) for the design.

## Tests

```bash
pytest            # vector/hybrid tests auto-skip if chromadb isn't installed
                  # eval/ IR metrics are covered offline (no Ragas/LLM needed)
```

## Layout

```
rag/config.py          settings (RAG_* env)
rag/models.py          Document, Page, Chunk, TreeNode, RetrievedContext, Answer
rag/documents/         PDF/Markdown/text loader
rag/embeddings/        local (sentence-transformers) + openai-compatible
rag/llm/               one OpenAI-compatible provider + structured-output fallback
rag/retrievers/        vector, pageindex, hybrid
rag/generation/        answer synthesis with citations
rag/pipeline.py        ingest + ask
app/api.py             FastAPI service
web/                   React + Vite single-page UI
eval/                  retrieval-mode benchmark (deterministic IR + Ragas)
```
