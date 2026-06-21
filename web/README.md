# RAG System — Web UI

A small React + Vite single-page app for the RAG System API: ingest documents,
pick a retrieval mode (vector / pageindex / hybrid), ask questions, and see
answers with their source citations.

The **⚙ settings panel** (top-right) configures the generation, reasoning, and
embedding endpoints — base_url / model / api_key for any OpenAI-compatible server,
with one-click presets (Ollama, OpenAI, OpenRouter). Changes apply immediately
(via `PUT /config`) and, by default, are written back to the backend's `.env` so
they survive a restart. Switching the embedding model warns you to re-ingest.

## Run

```bash
# 1. Start the backend (from the repo root)
uvicorn app.api:app --reload          # serves on :8000

# 2. Start the UI (from web/)
cd web
npm install
npm run dev                           # serves on http://localhost:5173
```

In dev the Vite proxy forwards `/health`, `/ingest` and `/ask` to the backend on
`:8000`, so there is no CORS to configure. If your API is elsewhere:

```bash
VITE_API_TARGET=http://my-host:9000 npm run dev
```

## Production build

```bash
npm run build        # outputs static assets to web/dist/
npm run preview      # serve the build locally to check it
```

A production build talks to the API via `VITE_API_BASE_URL` (see `.env.example`).
Set it to the API origin, e.g.:

```bash
echo 'VITE_API_BASE_URL=http://localhost:8000' > .env
npm run build
```

The backend already sends permissive CORS headers, so a separately-hosted build
can call it directly.

## Layout

```
web/
  index.html             app entry
  vite.config.js         dev server + API proxy
  src/
    main.jsx             React root
    App.jsx              state: health, mode, top_k, messages
    api.js               fetch client for /health, /ingest, /ask
    styles.css
    components/
      Header.jsx         title, health badge, mode + top_k controls
      IngestBar.jsx      drag/drop upload + ingest-by-path
      Chat.jsx           message list + composer
      Message.jsx        a bubble + its citations
```
