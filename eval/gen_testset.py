"""Synthesize an evaluation test set — lightweight, page-grounded, GB10-gentle.

Why not Ragas' ``TestsetGenerator``: it builds a knowledge graph first
(Headlines/Summary/NodeFilter/Embedding/Themes/NER over every node) firing
hundreds of *highly concurrent* LLM + embedding calls before writing a single
question — ~80 min on a single local Ollama and enough sustained load to
overheat the GB10. This generator instead makes **one sequential LLM call per
sampled page**: pick N substantive pages spread across the corpus and ask the
model for one self-contained question + reference answer grounded in that page.
Gold labels are then the *exact* ``(doc_id, page)`` — no fuzzy matching.

Gentle on the box by construction: concurrency 1, retry-with-backoff, optional
``--pace`` sleep between calls, and an early stop after repeated connection
failures (so we never hammer a struggling endpoint).

Output: ``eval/testset.jsonl``, one JSON object per line::

    {question, reference_answer, reference_contexts, gold_doc_id, gold_pages, synthesizer}

Usage::

    python -m eval.gen_testset --n 12
    python -m eval.gen_testset --n 12 --corpus <files...> --pace 1.0
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from rag.documents.loader import load_documents
from rag.llm.factory import build_llm

from eval.settings import load_settings

_QA_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["question", "answer"],
}

_SYSTEM = (
    "You write evaluation questions for a document-retrieval system over clinical "
    "and regulatory texts (TB guidelines and drug labels). Given the text of ONE "
    "page, write exactly ONE specific, self-contained question answerable using "
    "ONLY that page, together with a short correct answer grounded in the page. "
    "The question must stand alone (never refer to 'this page/passage/table/"
    "document'), and must name the drug or topic so it is findable. Prefer "
    "concrete facts (doses, adverse reactions, definitions, thresholds). Avoid "
    "yes/no questions."
)

# Drop questions that leak their source instead of standing alone.
_BAD_QUESTION = re.compile(
    r"\b(this|the\s+(above|following|given|preceding))\s+"
    r"(page|passage|text|table|document|section|excerpt|content)\b"
    r"|according to the\s+(text|passage|page|document)",
    re.IGNORECASE,
)


def _candidate_pages(docs, min_chars: int) -> dict[str, list[tuple]]:
    """Map doc_id -> [(doc_id, filename, page_number, text)] for prose-rich pages."""
    by_doc: dict[str, list[tuple]] = {}
    for d in docs:
        fname = d.metadata.get("filename", d.id)
        kept = [
            (d.id, fname, p.number, p.text)
            for p in d.pages
            if len(p.text.strip()) >= min_chars
        ]
        if kept:
            by_doc[d.id] = kept
    return by_doc


def _select_pages(by_doc: dict[str, list[tuple]], n: int, seed: int) -> list[tuple]:
    """Round-robin across docs (longest pages first) for even corpus coverage."""
    ordered = {
        doc_id: sorted(pages, key=lambda t: len(t[3]), reverse=True)
        for doc_id, pages in by_doc.items()
    }
    doc_ids = list(ordered.keys())
    random.Random(seed).shuffle(doc_ids)
    total = sum(len(v) for v in ordered.values())
    n = min(n, total)
    selected: list[tuple] = []
    i = 0
    while len(selected) < n:
        bucket = ordered[doc_ids[i % len(doc_ids)]]
        if bucket:
            selected.append(bucket.pop(0))
        i += 1
    return selected


def _generate_one(llm, fname: str, page_no: int, text: str, max_chars: int) -> dict:
    prompt = (
        f"Document: {fname}\nPage {page_no}\n"
        f'Page text:\n"""\n{text.strip()[:max_chars]}\n"""\n\n'
        "Return a JSON object with keys 'question' and 'answer'."
    )
    return llm.structured(prompt, _QA_SCHEMA, system=_SYSTEM, schema_name="qa")


def main() -> None:
    ap = argparse.ArgumentParser(description="Page-grounded eval test-set generator.")
    ap.add_argument("--n", type=int, default=None, help="number of questions")
    ap.add_argument(
        "--corpus",
        nargs="+",
        default=None,
        help="corpus dir(s) or explicit file list (default: RAG_EVAL_CORPUS_DIR). "
        "Pass the same files you ingested so gold labels match the index.",
    )
    ap.add_argument("--out", default=None, help="output jsonl path")
    ap.add_argument("--min-chars", type=int, default=400, help="skip pages shorter than this")
    ap.add_argument("--max-chars", type=int, default=4000, help="page text sent to the LLM")
    ap.add_argument("--pace", type=float, default=0.0, help="seconds to sleep between calls")
    ap.add_argument("--seed", type=int, default=0, help="page-selection shuffle seed")
    ap.add_argument("--max-retries", type=int, default=3, help="retries per page on error")
    args = ap.parse_args()

    rag_settings, eval_settings = load_settings()
    n = args.n or eval_settings.n
    corpus = args.corpus or [eval_settings.corpus_dir]
    out_path = Path(args.out or eval_settings.testset_path)

    llm = build_llm("generation", rag_settings)
    if not llm.ping():
        raise SystemExit(
            f"LLM endpoint unreachable ({rag_settings.generation_base_url}). Is the GB10 on?"
        )

    docs = load_documents(corpus)
    by_doc = _candidate_pages(docs, args.min_chars)
    if not by_doc:
        raise SystemExit(f"No pages with >= {args.min_chars} chars; lower --min-chars.")
    pages = _select_pages(by_doc, n, args.seed)
    n_cand = sum(len(v) for v in by_doc.values())
    print(
        f"[gen] {len(docs)} docs, {n_cand} candidate pages; "
        f"generating {len(pages)} questions sequentially (concurrency 1)…",
        flush=True,
    )

    rows: list[dict] = []
    consecutive_fail = 0
    for i, (doc_id, fname, page_no, text) in enumerate(pages, 1):
        data = None
        for attempt in range(1, args.max_retries + 1):
            try:
                data = _generate_one(llm, fname, page_no, text, args.max_chars)
                break
            except Exception as e:
                wait = 2**attempt
                print(
                    f"[{i}/{len(pages)}] attempt {attempt}/{args.max_retries} failed: "
                    f"{e!r}; retry in {wait}s",
                    flush=True,
                )
                time.sleep(wait)

        if data is None:
            consecutive_fail += 1
            print(f"[{i}/{len(pages)}] gave up on {fname} p.{page_no}", flush=True)
            if consecutive_fail >= 3:
                print(
                    "[gen] 3 consecutive failures — stopping early to spare the endpoint "
                    "(check the GB10).",
                    flush=True,
                )
                break
            continue
        consecutive_fail = 0

        q = str(data.get("question", "")).strip()
        a = str(data.get("answer", "")).strip()
        if len(q) < 12 or len(a) < 3 or _BAD_QUESTION.search(q):
            print(f"[{i}/{len(pages)}] skip weak Q  {fname} p.{page_no}", flush=True)
        else:
            rows.append(
                {
                    "question": q,
                    "reference_answer": a,
                    "reference_contexts": [text.strip()[: args.max_chars]],
                    "gold_doc_id": doc_id,
                    "gold_pages": [page_no],
                    "synthesizer": "page-grounded",
                }
            )
            print(f"[{i}/{len(pages)}] OK  {fname} p.{page_no}  Q: {q[:70]}", flush=True)

        if args.pace:
            time.sleep(args.pace)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[gen] wrote {len(rows)} questions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
