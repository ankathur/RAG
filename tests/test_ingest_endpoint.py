"""Regression tests for /ingest request routing (no LLM / GB10 needed).

Guards the bug where mixing a multipart ``File`` param with a JSON body model
made FastAPI expect multipart and silently ignore JSON ``{"paths": [...]}``
bodies — breaking path/folder ingest.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from app import api  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    calls: list = []
    # Stub the pipeline so we exercise routing only — never touch the LLM/embedder.
    monkeypatch.setattr(
        api.pipeline, "ingest", lambda paths: calls.append(paths) or {"ingested": paths}
    )
    c = TestClient(api.app)
    c.calls = calls  # type: ignore[attr-defined]
    return c


def test_json_paths_route_to_ingest(client):
    r = client.post("/ingest", json={"paths": ["data/kb/tb", "x.md"]})
    assert r.status_code == 200, r.text
    assert client.calls == [["data/kb/tb", "x.md"]]  # folder path passed through


def test_json_missing_paths_is_bad_request(client):
    r = client.post("/ingest", json={"nope": 1})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "bad_request"
    assert client.calls == []


def test_multipart_file_routes_to_ingest(client):
    files = {"file": ("note.md", io.BytesIO(b"# hi\n\nbody"), "text/markdown")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 200, r.text
    assert len(client.calls) == 1  # called once with the saved temp-file path


def test_multipart_unsupported_ext(client):
    files = {"file": ("bad.xyz", io.BytesIO(b"nope"), "application/octet-stream")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "unsupported_media_type"
    assert client.calls == []
