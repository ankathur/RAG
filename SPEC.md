# RAG System — Specification

## 1. Overview

A domain-agnostic Retrieval-Augmented Generation (RAG) service. The knowledge base and domain
are supplied later; the system is generic and configurable. It exposes three interchangeable
retrieval strategies behind one interface and serves them over a REST API.

- **Vector** — chunk → embed → similarity search.
- **PageIndex** — vectorless, reasoning-based retrieval: build a hierarchical table-of-contents
  tree per document, then have the LLM reason / tree-search over it to select relevant sections.
- **Hybrid** — run both, merge, LLM-rerank.

The LLM is reached through a **single OpenAI-compatible client**: any model at an
OpenAI-compatible `/v1` endpoint works via `base_url` + `api_key` + `model` (Ollama, LM Studio,
vLLM, llama.cpp, OpenAI, OpenRouter, Together, Groq, Claude-compat, …). Default is **local
Ollama**, so the system runs fully offline with no API keys.

### 1.1 Goals
- One common retrieval interface; three swappable backends selectable globally or per request.
- Provider-neutral LLM access (OpenAI-compatible), configurable per role (reasoning vs generation).
- Local-first default (no keys, offline); cloud endpoints drop in via env only.
- Ingest PDF, Markdown, and plain text; answers carry source citations (doc + page/section).

### 1.2 Non-goals (v1)
- Authentication/multi-tenant access control on the API.
- Distributed/streaming-to-client token output (responses returned whole).
- OCR for scanned PDFs; incremental re-embedding diffing beyond per-document rebuild.

## 2. Glossary
- **Chunk** — fixed-size text span used by vector mode.
- **TreeNode** — a node in the PageIndex table-of-contents tree (section, summary, page range).
- **RetrievedContext** — a unit of retrieved evidence passed to generation, with its source.
- **Role** — `reasoning` (tree build/search, rerank) or `generation` (answer synthesis).

## 3. Architecture

```
app/api.py  ──>  rag/pipeline.py  ──>  rag/factory.py ─> Retriever (vector|pageindex|hybrid)
                                   └─>  rag/generation/generator.py
Retriever & generator use:
  rag/llm/*           (OpenAI-compatible provider, per role)
  rag/embeddings/*    (local sentence-transformers default; optional /v1/embeddings)
  rag/documents/*     (PDF/MD/txt loader)
Persistence: data/indexes/{chroma/, pageindex/*.json}
```

Module map:
```
rag/config.py                # Settings (pydantic-settings)
rag/models.py                # Document, Page, Chunk, TreeNode, RetrievedContext, Answer
rag/documents/loader.py      # load_documents(paths) -> list[Document]
rag/embeddings/{base,local,openai_compat}.py
rag/llm/{base,openai_compat,structured,factory}.py
rag/retrievers/{base,vector,pageindex,hybrid}.py
rag/generation/generator.py
rag/pipeline.py              # RAGPipeline
rag/factory.py               # build_retriever(mode, settings)
app/api.py                   # FastAPI
```

## 4. Configuration (env, prefix `RAG_`)

| Setting | Default | Purpose |
|---|---|---|
| `retrieval_mode` | `hybrid` | Default mode: `vector` \| `pageindex` \| `hybrid` |
| `top_k` | `6` | Contexts returned to generation |
| `hybrid_strategy` | `merge_rerank` | `merge_rerank` \| `pageindex_then_vector` |
| `reasoning_base_url` | `http://localhost:11434/v1` | OpenAI-compatible endpoint for reasoning |
| `reasoning_api_key` | `ollama` | Key for that endpoint (dummy ok for local) |
| `reasoning_model` | `qwen2.5:7b-instruct` | Reasoning model id |
| `generation_base_url` | `http://localhost:11434/v1` | Endpoint for answer synthesis |
| `generation_api_key` | `ollama` | Key |
| `generation_model` | `qwen2.5:7b-instruct` | Generation model id |
| `structured_output_mode` | `auto` | `auto` \| `json_schema` \| `json_object` \| `prompt` |
| `embedding_provider` | `local` | `local` \| `openai` |
| `embedding_model` | `all-MiniLM-L6-v2` | Embedding model id |
| `embedding_base_url` / `embedding_api_key` | unset | For `embedding_provider=openai` (`/v1/embeddings`) |
| `chunk_size` / `chunk_overlap` | `800` / `120` | Vector chunking (chars) |
| `chroma_dir` | `data/indexes/chroma` | Persistent vector store path |
| `index_dir` | `data/indexes/pageindex` | PageIndex tree JSON path |
| `pageindex_max_pages_per_node` | `10` | Tree build granularity |

## 5. Data models (`rag/models.py`, pydantic)

- `Page{ number:int, text:str }`
- `Document{ id:str, source_path:str, format:"pdf"|"md"|"txt", pages:list[Page], metadata:dict }`
- `Chunk{ id:str, doc_id:str, text:str, page:int|None, position:int }`
- `TreeNode{ node_id:str, title:str, summary:str, start_page:int, end_page:int, nodes:list[TreeNode] }`
- `RetrievedContext{ text:str, doc_id:str, locator:str, score:float|None, origin:"vector"|"pageindex" }`
  (`locator` = e.g. `"p.12"` or `"§2.3 Risk Factors"`)
- `Answer{ text:str, citations:list[RetrievedContext], mode:str, usage:dict }`

## 6. LLM provider (`rag/llm/`)

- `LLMProvider` interface: `complete(prompt, *, system=None) -> str`,
  `stream(prompt, *, system=None) -> Iterator[str]`,
  `structured(prompt, schema, *, system=None) -> dict`.
- `openai_compat.py`: wraps the `openai` SDK (`OpenAI(base_url, api_key)` →
  `chat.completions.create(model, …)`). One implementation for all endpoints.
- `structured.py`: per `structured_output_mode`, attempt in order and fall back —
  `response_format={"type":"json_schema","strict":true,...}` → `{"type":"json_object"}` →
  prompt-instructed JSON; then parse tolerantly and, on failure, one repair retry. Returns a
  validated dict matching the requested schema.
- `factory.build_llm(role, settings)` returns a provider configured from the role's
  `*_base_url/_api_key/_model`. Reasoning and generation are independent.

## 7. Embeddings (`rag/embeddings/`)

- `EmbeddingProvider`: `embed_documents(list[str]) -> list[list[float]]`,
  `embed_query(str) -> list[float]`, `dim:int`.
- `local.py`: `sentence-transformers` (default `all-MiniLM-L6-v2`).
- `openai_compat.py`: `/v1/embeddings` via the `openai` SDK when `embedding_provider=openai`.

## 8. Ingestion (`rag/documents/loader.py`)

- `load_documents(paths) -> list[Document]`. Dispatch by extension:
  - `.pdf` → `pypdf`; one `Page` per page; also read `reader.outline` (bookmarks) when present.
  - `.md` / `.txt` → split on headings (md) or fixed windows; synthesize `Page`s with offsets.
- Deterministic `Document.id` from path + content hash (for cache invalidation).

## 9. Retrieval modes (`rag/retrievers/`)

`Retriever` ABC: `index(docs)`, `retrieve(query, k) -> list[RetrievedContext]`, `persist()`, `load()`.

### 9.1 Vector (`vector.py`)
Chunk → `embed_documents` → upsert into a persistent Chroma collection with `{doc_id, page}`
metadata. `retrieve` = `embed_query` → top-k similarity → `RetrievedContext(origin="vector")`.

### 9.2 PageIndex (`pageindex.py`)
- **Build:** for each document, LLM `structured()` call(s) produce a `TreeNode` tree (title,
  summary, page range, children). Long docs built section-by-section then stitched; when a PDF
  outline exists, seed structure from it and have the LLM only summarize. Persist to
  `index_dir/<doc_id>.json`. Raw page text remains addressable by page range.
- **Search (retrieval):** serialize the tree (`node_id · title · summary`, indented); LLM
  `structured()` returns `{ selected_node_ids:[...], rationale:str }`. Fetch raw text for those
  nodes' page ranges → `RetrievedContext(origin="pageindex", locator="p.<range>/§<title>")`.
  v1 = single-shot whole-tree reasoning; hook for iterative depth-descent on very large trees.

### 9.3 Hybrid (`hybrid.py`)
- `merge_rerank` (default): run vector + pageindex, dedupe overlapping contexts, LLM rerank
  (`structured()` → ordered kept ids), return top-k.
- `pageindex_then_vector`: PageIndex selects sections; vector search restricted to those pages.

## 10. Generation (`rag/generation/generator.py`)
`answer(query, contexts, mode) -> Answer`. Build a system+user prompt embedding labeled
contexts; instruct the model to answer **only** from context and cite sources by locator;
stream via the generation provider; assemble `Answer` with the cited contexts and token usage.

## 11. Pipeline & factory
- `factory.build_retriever(mode, settings)` → the chosen `Retriever`.
- `RAGPipeline(retriever, generator)`: `ingest(paths)` → `retriever.index`;
  `ask(query, mode?) -> Answer` → `retriever.retrieve` → `generator.answer`.

## 12. REST API (`app/api.py`)

- `GET /health` → `200 {"status":"ok","llm_reachable":bool,"mode":<default>}`.
- `POST /ingest` (multipart file **or** `{"paths":[...]}`) → `200 {"ingested":[doc_id...],
  "vector":bool,"pageindex":bool}`. Indexes for all enabled modes are built/persisted.
- `POST /ask` `{"query":str, "mode"?:"vector"|"pageindex"|"hybrid", "top_k"?:int}` →
  `200 {"answer":str,"citations":[{doc_id,locator,origin,score}],"mode":str,"usage":{...}}`.
- `GET /config` → `200` current LLM/embedding config with **API keys masked** (each role
  reports `api_key_set:bool`, never the value): `{"reasoning":{base_url,model,api_key_set},
  "generation":{...},"embedding":{provider,model,base_url,api_key_set},structured_output_mode}`.
- `PUT /config` (all fields optional) updates the LLM/embedding endpoints **at runtime** without
  a restart: `{"reasoning"?:{base_url?,api_key?,model?}, "generation"?:{...},
  "embedding"?:{provider?,model?,base_url?,api_key?}, "structured_output_mode"?, "persist"?:bool}`.
  A blank/omitted `api_key` keeps the current key. The change is applied to the live pipeline
  (clients rebuilt; persisted indexes untouched) and, when `persist` is true (default), written
  back to `.env` so it survives a restart. Returns the masked config plus
  `{"llm_reachable":bool,"persisted":bool,"embedding_changed":bool}`. `embedding_changed:true`
  signals that documents must be re-ingested (the vector index no longer matches the embeddings).
- CORS is permissive (`*`) so a browser SPA on another origin can call the API.
- Errors: `400` (bad input/unsupported format), `422` (validation), `503` (LLM endpoint
  unreachable), `500` (unexpected). Body: `{"error":{"type":str,"message":str}}`.
- Startup loads persisted indexes; `mode` defaults to `retrieval_mode`, overridable per request.

## 13. Persistence
- Vector: Chroma persistent client at `chroma_dir` (one collection per knowledge base).
- PageIndex: one JSON tree per document at `index_dir/<doc_id>.json`.
- Rebuild a document's index only when its content hash changes.

## 14. Error handling & edge cases
- Empty/again-asked query → `400`. Unsupported file type → `400` listing accepted types.
- LLM endpoint down → `503` with the configured `base_url` named.
- Structured-output failure after fallback+repair → surfaced as `500` with the raw text logged.
- No contexts retrieved → generator returns an explicit "no relevant context found" answer, not a hallucination.

## 15. Security & privacy
- Local default keeps documents and queries on-machine; no external calls unless an endpoint is configured.
- Sanitize uploaded filenames (`os.path.basename`) before writing; write under `data/` only.
- API keys read from env, never logged; no secrets in tree JSON or citations.

## 16. Performance
- Tree build runs the LLM at ingest time (cached on disk). Vector ingest is embedding-bound.
- `top_k` and `pageindex_max_pages_per_node` bound context size and token spend.

## 17. Testing & acceptance criteria
- `tests/test_loader.py`: PDF/MD/txt parse into `Document` with correct pages.
- `tests/test_pageindex_tree.py`: tree build on a tiny doc yields a well-formed `TreeNode` tree;
  search returns ≥1 plausible node (LLM mocked).
- `tests/test_pipeline_smoke.py`: end-to-end `ingest`→`ask` for each mode with mocked LLM.
- **Acceptance:** all three modes answer the sample question with citations pointing at real
  pages/sections; swapping `*_base_url`/`*_model` to a second endpoint works with no code change.

## 18. Dependencies & prerequisites
`openai`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic`, `pydantic-settings`,
`pypdf`, `chromadb`, `sentence-transformers`. Prereq: an OpenAI-compatible endpoint (default
Ollama: `ollama serve` + `ollama pull qwen2.5:7b-instruct`). Use a Python 3.12 venv (3.14 wheel
lag for torch/chromadb).

## 19. Future enhancements
- Streaming token responses to the client; auth; reranker model for vector; OCR for scanned PDFs;
  iterative tree-descent search; per-tenant knowledge bases; evaluation harness for mode comparison.
