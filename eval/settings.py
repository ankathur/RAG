"""Evaluation configuration (env-driven, prefix ``RAG_EVAL_``).

Reuses :class:`rag.config.Settings` for the *pipeline* side (the system under
test). This file only adds knobs specific to evaluation: the independent judge
model, the embeddings used by Ragas, and run sizing. Defaults target the GB10
Ollama stack described in the project memory.

The Ragas/LangChain client builders import lazily so that ``import
eval.settings`` (and the offline metric tests) work without the eval extras
installed.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.config import Settings, get_settings


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Judge LLM (independent family from the generator → less self-grading) ----
    judge_model: str = "gemma4:26b"
    judge_base_url: str | None = None  # defaults to the generation endpoint
    judge_api_key: str | None = None

    # ---- Embeddings for Ragas (test-set synthesis + answer relevancy) ----
    # Default to the pipeline's embedding config (bge-m3 on the GB10).
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None

    # ---- Run sizing ----
    n: int = 25  # questions to synthesize / evaluate
    k: int | None = None  # retrieval depth; defaults to Settings.top_k
    corpus_dir: str = "data/kb/tb"
    testset_path: str = "eval/testset.jsonl"
    output_dir: str = "eval/results"
    request_timeout: float = 600.0  # generous for slow local generations
    # Ragas evaluation concurrency. Keep at 1 (sequential) so we never overwhelm
    # / overheat a single local Ollama box; raise only for a beefier endpoint.
    ragas_max_workers: int = 1

    # -- resolution helpers ---------------------------------------------------
    def retrieval_k(self, rag: Settings) -> int:
        return self.k or rag.top_k

    def _judge_endpoint(self, rag: Settings) -> tuple[str, str, str]:
        base = self.judge_base_url or rag.generation_base_url
        key = self.judge_api_key or rag.generation_api_key
        return base, key, self.judge_model

    def _embedding_endpoint(self, rag: Settings) -> tuple[str, str, str]:
        base = self.embedding_base_url or rag.embedding_base_url or rag.generation_base_url
        key = self.embedding_api_key or rag.embedding_api_key or rag.generation_api_key
        model = self.embedding_model or rag.embedding_model
        return base, key, model

    # -- Ragas client builders (lazy imports) ---------------------------------
    def build_judge_llm(self, rag: Settings):
        """Ragas-wrapped chat model used as the LLM judge / test generator."""
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        base, key, model = self._judge_endpoint(rag)
        return LangchainLLMWrapper(
            ChatOpenAI(
                model=model,
                base_url=base,
                api_key=key,
                temperature=0,
                timeout=self.request_timeout,
                max_retries=2,
            )
        )

    def build_embeddings(self, rag: Settings):
        """Ragas-wrapped embeddings (bge-m3 by default) via OpenAI-compatible API."""
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        base, key, model = self._embedding_endpoint(rag)
        return LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=model,
                base_url=base,
                api_key=key,
                # bge-m3 (and other non-OpenAI models) aren't tiktoken-tokenizable;
                # skip the length check so embedding calls go straight through.
                check_embedding_ctx_length=False,
            )
        )


def load_settings() -> tuple[Settings, EvalSettings]:
    """Return the (pipeline, eval) settings pair used across the harness."""
    return get_settings(), EvalSettings()
