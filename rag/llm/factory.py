"""Build LLM providers per role (see SPEC.md §6)."""

from __future__ import annotations

from typing import Literal

from rag.config import Settings, get_settings
from rag.llm.base import LLMProvider

Role = Literal["reasoning", "generation"]


def build_llm(role: Role, settings: Settings | None = None) -> LLMProvider:
    """Return an LLM provider configured for ``role`` from settings.

    Reasoning and generation are independent endpoints, so you can run, e.g.,
    a strong cloud model for tree-search and a local model for synthesis.
    """
    settings = settings or get_settings()
    base_url, api_key, model = settings.llm_for(role)

    from rag.llm.openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(
        base_url=base_url,
        api_key=api_key,
        model=model,
        structured_mode=settings.structured_output_mode,
    )
