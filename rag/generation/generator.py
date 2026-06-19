"""Answer synthesis from retrieved context (see SPEC.md §10)."""

from __future__ import annotations

from rag.config import Settings, get_settings
from rag.llm.base import LLMProvider
from rag.llm.factory import build_llm
from rag.models import Answer, RetrievedContext

_SYSTEM = (
    "You are a careful research assistant. Answer the user's question using ONLY "
    "the provided context passages. Cite the sources you use inline with their "
    "labels like [S1], [S2]. If the context does not contain the answer, say so "
    "plainly and do not invent facts."
)

_NO_CONTEXT = (
    "I couldn't find relevant information in the knowledge base to answer that."
)


class Generator:
    def __init__(
        self, settings: Settings | None = None, llm: LLMProvider | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = build_llm("generation", self.settings)
        return self._llm

    def answer(
        self, query: str, contexts: list[RetrievedContext], mode: str
    ) -> Answer:
        if not contexts:
            return Answer(text=_NO_CONTEXT, citations=[], mode=mode)

        blocks = []
        for i, ctx in enumerate(contexts, start=1):
            label = f"S{i}"
            blocks.append(
                f"[{label}] (doc={ctx.doc_id}, {ctx.locator}, via {ctx.origin})\n"
                f"{ctx.text.strip()}"
            )
        context_block = "\n\n".join(blocks)
        prompt = (
            f"Question: {query}\n\n"
            f"Context passages:\n{context_block}\n\n"
            "Write a concise, well-grounded answer. Use inline citations like "
            "[S1] that refer to the passages above."
        )

        text_parts = list(self.llm.stream(prompt, system=_SYSTEM))
        text = "".join(text_parts).strip()
        if not text:  # some endpoints don't stream; fall back to a full call
            text = self.llm.complete(prompt, system=_SYSTEM).strip()

        return Answer(
            text=text or _NO_CONTEXT,
            citations=contexts,
            mode=mode,
            usage={"contexts": len(contexts)},
        )
