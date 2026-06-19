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

## Switching the model / endpoint

Everything is config (`.env`, prefix `RAG_`). Point any role at any OpenAI-compatible endpoint
without code changes — e.g. run reasoning on a stronger remote model, generation locally:

```bash
RAG_REASONING_BASE_URL=https://openrouter.ai/api/v1
RAG_REASONING_API_KEY=sk-...
RAG_REASONING_MODEL=meta-llama/llama-3.1-70b-instruct
# generation_* left at the local Ollama default
```

## Tests

```bash
pytest            # vector/hybrid tests auto-skip if chromadb isn't installed
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
```
