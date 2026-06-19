"""FastAPI service exposing the RAG pipeline (see SPEC.md §12).

Run with:  uvicorn app.api:app --reload
"""

from __future__ import annotations

import os
import tempfile
from typing import Literal

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag.config import get_settings
from rag.documents.loader import SUPPORTED_EXTS
from rag.llm.base import LLMError, LLMUnreachableError
from rag.llm.factory import build_llm
from rag.pipeline import RAGPipeline

settings = get_settings()
app = FastAPI(title="RAG System", version="0.1.0")
pipeline = RAGPipeline(settings)

Mode = Literal["vector", "pageindex", "hybrid"]


class IngestPaths(BaseModel):
    paths: list[str]


class AskRequest(BaseModel):
    query: str
    mode: Mode | None = None
    top_k: int | None = None


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
async def ingest(
    file: UploadFile | None = File(default=None),
    paths: IngestPaths | None = None,
):
    try:
        if file is not None:
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in SUPPORTED_EXTS:
                return _error(
                    400,
                    "unsupported_media_type",
                    f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED_EXTS)}",
                )
            safe_name = os.path.basename(file.filename or f"upload{ext}")
            with tempfile.TemporaryDirectory() as tmp:
                dest = os.path.join(tmp, safe_name)
                with open(dest, "wb") as fh:
                    fh.write(await file.read())
                result = pipeline.ingest(dest)
        elif paths is not None and paths.paths:
            result = pipeline.ingest(paths.paths)
        else:
            return _error(400, "bad_request", "Provide a file upload or {'paths': [...]}.")
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
