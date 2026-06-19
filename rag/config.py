"""Application configuration (see SPEC.md §4).

All settings are read from the environment with the ``RAG_`` prefix, or from a
local ``.env`` file. Defaults make the system run fully offline against a local
Ollama server with no real API keys.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["vector", "pageindex", "hybrid"]
HybridStrategy = Literal["merge_rerank", "pageindex_then_vector"]
StructuredMode = Literal["auto", "json_schema", "json_object", "prompt"]
EmbeddingProvider = Literal["local", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Retrieval ----
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 6
    hybrid_strategy: HybridStrategy = "merge_rerank"

    # ---- LLM: reasoning role (tree build/search, rerank) ----
    reasoning_base_url: str = "http://localhost:11434/v1"
    reasoning_api_key: str = "ollama"
    reasoning_model: str = "qwen2.5:7b-instruct"

    # ---- LLM: generation role (answer synthesis) ----
    generation_base_url: str = "http://localhost:11434/v1"
    generation_api_key: str = "ollama"
    generation_model: str = "qwen2.5:7b-instruct"

    structured_output_mode: StructuredMode = "auto"

    # ---- Embeddings ----
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None

    # ---- Chunking (vector) ----
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- PageIndex ----
    pageindex_max_pages_per_node: int = 10

    # ---- Persistence ----
    chroma_dir: str = "data/indexes/chroma"
    index_dir: str = "data/indexes/pageindex"

    def llm_for(self, role: Literal["reasoning", "generation"]) -> tuple[str, str, str]:
        """Return ``(base_url, api_key, model)`` for the given LLM role."""
        if role == "reasoning":
            return self.reasoning_base_url, self.reasoning_api_key, self.reasoning_model
        return self.generation_base_url, self.generation_api_key, self.generation_model


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so the whole app shares one configuration."""
    return Settings()
