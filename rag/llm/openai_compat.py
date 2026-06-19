"""A single LLM provider over any OpenAI-compatible endpoint (see SPEC.md §6)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from rag.llm.base import LLMProvider, LLMUnreachableError
from rag.llm.structured import request_structured


class OpenAICompatLLM(LLMProvider):
    """Talks to one OpenAI-compatible ``/v1`` endpoint via the ``openai`` SDK.

    Works unchanged against Ollama, LM Studio, vLLM, llama.cpp, OpenAI,
    OpenRouter, Together, Groq, Claude-compat, etc. — only the constructor
    arguments change.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        structured_mode: str = "auto",
        timeout: float = 120.0,
    ) -> None:
        # Lazy import so the package imports even if `openai` is absent (tests mock).
        from openai import OpenAI

        self.model = model
        self._structured_mode = structured_mode
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._base_url = base_url

    # -- helpers ----------------------------------------------------------------
    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _wrap_conn_errors(self, exc: Exception) -> Exception:
        # openai.APIConnectionError / APITimeoutError -> our 503-mappable error.
        name = type(exc).__name__
        if "Connection" in name or "Timeout" in name:
            return LLMUnreachableError(
                f"LLM endpoint {self._base_url!r} is unreachable ({name})."
            )
        return exc

    # -- LLMProvider ------------------------------------------------------------
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt, system),
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001 - re-raise as typed error
            raise self._wrap_conn_errors(exc) from exc
        return resp.choices[0].message.content or ""

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt, system),
                temperature=0,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_conn_errors(exc) from exc

    def structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str | None = None,
        schema_name: str = "result",
    ) -> dict[str, Any]:
        try:
            return request_structured(
                client=self._client,
                model=self.model,
                prompt=prompt,
                schema=schema,
                mode=self._structured_mode,
                system=system,
                schema_name=schema_name,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_conn_errors(exc) from exc

    def ping(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:  # noqa: BLE001
            return False
