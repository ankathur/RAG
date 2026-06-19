"""LLM provider interface and shared errors (see SPEC.md §6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class LLMError(RuntimeError):
    """Base class for LLM provider failures."""


class LLMUnreachableError(LLMError):
    """The configured OpenAI-compatible endpoint could not be reached."""


class StructuredOutputError(LLMError):
    """The model could not produce valid JSON for the requested schema."""


class LLMProvider(ABC):
    """A provider for a single LLM role (reasoning or generation).

    Implementations talk to one OpenAI-compatible endpoint configured by
    ``base_url`` / ``api_key`` / ``model``.
    """

    model: str

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the full text completion for ``prompt``."""

    @abstractmethod
    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        """Yield text deltas for ``prompt``."""

    @abstractmethod
    def structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str | None = None,
        schema_name: str = "result",
    ) -> dict[str, Any]:
        """Return a JSON object matching ``schema`` (JSON Schema dict)."""

    @abstractmethod
    def ping(self) -> bool:
        """Return True if the endpoint is reachable, False otherwise."""
