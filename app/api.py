"""FastAPI service exposing the RAG pipeline (see SPEC.md §12).

Run with:  uvicorn app.api:app --reload
"""

from __future__ import annotations

import os
import tempfile
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag.config import (
    EmbeddingProvider,
    RerankerProvider,
    StructuredMode,
    get_settings,
    write_env_vars,
)
from rag.documents.loader import SUPPORTED_EXTS
from rag.llm.base import LLMError, LLMUnreachableError
from rag.llm.factory import build_llm
from rag.pipeline import RAGPipeline

settings = get_settings()
app = FastAPI(title="RAG System", version="0.1.0")

# Permissive CORS so the React SPA (and any local-first client) can call the API
# from another origin. Tighten allow_origins for a hardened deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline(settings)

Mode = Literal["vector", "pageindex", "hybrid"]


class AskRequest(BaseModel):
    query: str
    mode: Mode | None = None
    top_k: int | None = None


class RoleConfig(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class EmbeddingConfig(BaseModel):
    provider: EmbeddingProvider | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class RerankerConfig(BaseModel):
    provider: RerankerProvider | None = None
    model: str | None = None
    top_k: int | None = None


class ConfigUpdate(BaseModel):
    reasoning: RoleConfig | None = None
    generation: RoleConfig | None = None
    embedding: EmbeddingConfig | None = None
    reranker: RerankerConfig | None = None
    structured_output_mode: StructuredMode | None = None
    persist: bool = True


def _error(status: int, type_: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"type": type_, "message": message}}
    )


@app.get("/health")
def health() -> dict:
    try:
        reachable = build_llm("generation", settings).ping()
    except Exception:
        reachable = False
    return {
        "status": "ok",
        "llm_reachable": reachable,
        "mode": settings.retrieval_mode,
    }


@app.post("/ingest")
async def ingest(request: Request):
    """Ingest a multipart file upload *or* a JSON ``{"paths": [...]}`` body.

    A single endpoint can't bind both an ``UploadFile`` (which forces
    ``multipart/form-data``) and a JSON model — FastAPI would ignore the JSON
    body — so we branch on the content type and parse the body ourselves.
    ``paths`` entries may be files or directories (directories are ingested
    recursively).
    """
    content_type = request.headers.get("content-type", "")
    try:
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            filename = getattr(upload, "filename", None)
            if not filename:
                return _error(400, "bad_request", "Provide a file upload or {'paths': [...]}.")
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTS:
                return _error(
                    400,
                    "unsupported_media_type",
                    f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED_EXTS)}",
                )
            safe_name = os.path.basename(filename) or f"upload{ext}"
            with tempfile.TemporaryDirectory() as tmp:
                dest = os.path.join(tmp, safe_name)
                with open(dest, "wb") as fh:
                    fh.write(await upload.read())
                result = pipeline.ingest(dest)
        else:
            try:
                body = await request.json()
            except Exception:
                body = None
            paths = body.get("paths") if isinstance(body, dict) else None
            if not paths:
                return _error(400, "bad_request", "Provide a file upload or {'paths': [...]}.")
            result = pipeline.ingest(paths)
    except FileNotFoundError as exc:
        return _error(400, "not_found", str(exc))
    except ValueError as exc:
        return _error(400, "bad_request", str(exc))
    except LLMUnreachableError as exc:
        return _error(503, "llm_unreachable", str(exc))
    except LLMError as exc:
        return _error(500, "llm_error", str(exc))
    return result


@app.post("/ask")
def ask(req: AskRequest):
    if not req.query.strip():
        return _error(400, "bad_request", "query must not be empty.")
    try:
        answer = pipeline.ask(req.query, mode=req.mode, top_k=req.top_k)
    except LLMUnreachableError as exc:
        return _error(503, "llm_unreachable", str(exc))
    except LLMError as exc:
        return _error(500, "llm_error", str(exc))
    return {
        "answer": answer.text,
        "citations": [
            {
                "doc_id": c.doc_id,
                "locator": c.locator,
                "origin": c.origin,
                "score": c.score,
            }
            for c in answer.citations
        ],
        "mode": answer.mode,
        "usage": answer.usage,
    }


def _current_config() -> dict:
    """Current LLM/embedding config with API keys masked (see SPEC.md §15)."""
    return {
        "reasoning": {
            "base_url": settings.reasoning_base_url,
            "model": settings.reasoning_model,
            "api_key_set": bool(settings.reasoning_api_key),
        },
        "generation": {
            "base_url": settings.generation_base_url,
            "model": settings.generation_model,
            "api_key_set": bool(settings.generation_api_key),
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "base_url": settings.embedding_base_url,
            "api_key_set": bool(settings.embedding_api_key),
        },
        "reranker": {
            "provider": settings.reranker,
            "model": settings.reranker_model,
            "top_k": settings.reranker_top_k,
        },
        "structured_output_mode": settings.structured_output_mode,
    }


def _apply_config(update: ConfigUpdate) -> tuple[dict[str, str], bool]:
    """Mutate the live settings from ``update``.

    Returns the changed ``RAG_*`` env vars (for optional persistence) and whether
    the embedding config changed (which invalidates the vector index).
    """
    env: dict[str, str] = {}

    def setf(field: str, value) -> None:
        # Blank / omitted string -> leave the existing value untouched (lets the
        # UI submit an empty api_key field to mean "keep current key").
        if value is None:
            return
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return
        setattr(settings, field, value)
        env[f"RAG_{field.upper()}"] = str(value)

    if update.reasoning:
        setf("reasoning_base_url", update.reasoning.base_url)
        setf("reasoning_api_key", update.reasoning.api_key)
        setf("reasoning_model", update.reasoning.model)
    if update.generation:
        setf("generation_base_url", update.generation.base_url)
        setf("generation_api_key", update.generation.api_key)
        setf("generation_model", update.generation.model)

    embedding_changed = False
    if update.embedding:
        before = (
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_base_url,
        )
        setf("embedding_provider", update.embedding.provider)
        setf("embedding_model", update.embedding.model)
        setf("embedding_base_url", update.embedding.base_url)
        setf("embedding_api_key", update.embedding.api_key)
        embedding_changed = before != (
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_base_url,
        )

    if update.reranker:
        setf("reranker", update.reranker.provider)
        setf("reranker_model", update.reranker.model)
        if update.reranker.top_k is not None:
            setf("reranker_top_k", update.reranker.top_k)

    if update.structured_output_mode:
        setf("structured_output_mode", update.structured_output_mode)

    return env, embedding_changed


@app.get("/config")
def get_config() -> dict:
    return _current_config()


@app.put("/config")
def update_config(update: ConfigUpdate):
    try:
        env, embedding_changed = _apply_config(update)
        pipeline.apply_settings()
        persisted = bool(update.persist and env)
        if persisted:
            write_env_vars(env)
    except Exception as exc:  # noqa: BLE001 - surface any failure as 500
        return _error(500, "config_error", str(exc))

    try:
        reachable = build_llm("generation", settings).ping()
    except Exception:
        reachable = False

    return {
        **_current_config(),
        "llm_reachable": reachable,
        "persisted": persisted,
        "embedding_changed": embedding_changed,
    }
