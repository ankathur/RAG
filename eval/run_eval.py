"""Compare retrieval modes on the synthesized test set.

For each ``mode in {vector, pageindex, hybrid}`` and each question:

1. call :meth:`rag.pipeline.RAGPipeline.ask` (fixed ``k`` across modes for a
   fair comparison), capturing the answer, its citations, and latency;
2. score retrieval deterministically (:mod:`eval.metrics`) against the gold
   doc/pages from the test set;
3. score RAG quality with Ragas (judge = ``gemma4``, embeddings = ``bge-m3``):
   context precision/recall, faithfulness, answer relevancy, factual correctness.

Outputs ``eval/results/<timestamp>/``: ``per_question.csv``, ``summary.csv``,
and ``summary.md`` (a modes × metrics table with a recommended mode).

Usage::

    python -m eval.run_eval --n 5            # smoke (3 modes × 5 questions)
    python -m eval.run_eval                   # full test set
    python -m eval.run_eval --modes vector hybrid --no-ragas

Run from the repo root (so ``.env`` and ``data/`` resolve). Heavy on the LLM
endpoint — consider running as a background job for large ``--n``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

from rag.pipeline import RAGPipeline

from eval.metrics import IR_METRICS, compute_ir, parse_locator
from eval.settings import EvalSettings, load_settings

ALL_MODES = ["vector", "pageindex", "hybrid"]
RAGAS_METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "factual_correctness",
]
# Order of metric columns reported everywhere.
SUMMARY_METRICS = [*IR_METRICS, *RAGAS_METRICS, "latency_s"]
_RAGAS_INPUT_COLS = {
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",
    "reference_contexts",
}


# -- test set -------------------------------------------------------------------
def load_testset(path: str, limit: int | None) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Test set not found: {path}. Run `python -m eval.gen_testset` first.")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


# -- ragas ----------------------------------------------------------------------
def _map_ragas_col(col: str) -> str:
    c = col.lower()
    if "context_precision" in c:
        return "context_precision"
    if "context_recall" in c:
        return "context_recall"
    if "faithful" in c:
        return "faithfulness"
    if "answer_relevancy" in c or "response_relevancy" in c:
        return "answer_relevancy"
    if "factual_correctness" in c:
        return "factual_correctness"
    return col


def _to_float(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def run_ragas(metas: list[dict], eval_settings: EvalSettings, rag_settings) -> list[dict]:
    """Score a mode's (question, answer, contexts, reference) tuples with Ragas."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
    except ImportError:  # older layout
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.evaluation import evaluate
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
        FactualCorrectness(),
    ]
    samples = [
        SingleTurnSample(
            user_input=m["question"],
            response=m["response"] or "",
            # faithfulness needs at least one context
            retrieved_contexts=m["contexts"] or [""],
            reference=m["reference"] or "",
        )
        for m in metas
    ]
    # max_workers=1 keeps Ragas fully sequential so we never overwhelm/overheat
    # a single local Ollama box (see EvalSettings.ragas_max_workers).
    run_config = RunConfig(
        max_workers=eval_settings.ragas_max_workers,
        timeout=int(eval_settings.request_timeout),
    )
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=eval_settings.build_judge_llm(rag_settings),
        embeddings=eval_settings.build_embeddings(rag_settings),
        run_config=run_config,
        raise_exceptions=False,
    )
    rdf = result.to_pandas()
    out: list[dict] = []
    for _, row in rdf.iterrows():
        scores: dict[str, float] = {}
        for col in rdf.columns:
            if col in _RAGAS_INPUT_COLS:
                continue
            scores[_map_ragas_col(col)] = _to_float(row[col])
        out.append(scores)
    return out


# -- aggregation / output -------------------------------------------------------
def _mean(values) -> float:
    xs = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return sum(xs) / len(xs) if xs else float("nan")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.3f}"


def write_outputs(
    per_question: list[dict],
    modes: list[str],
    k: int,
    eval_settings: EvalSettings,
    rag_settings,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode",
        "idx",
        "question",
        "gold_doc_id",
        "gold_pages",
        "retrieved_docs",
        "n_citations",
        "latency_s",
        *IR_METRICS,
        *RAGAS_METRICS,
        "error",
        "answer",
    ]
    with (out_dir / "per_question.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in per_question:
            w.writerow(row)

    # Per-mode means.
    summary = {
        mode: {
            metric: _mean([r.get(metric) for r in per_question if r["mode"] == mode])
            for metric in SUMMARY_METRICS
        }
        for mode in modes
    }
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", *SUMMARY_METRICS])
        for mode in modes:
            w.writerow([mode, *(summary[mode][m] for m in SUMMARY_METRICS)])

    ranked = sorted(
        modes,
        key=lambda m: (-_safe(summary[m]["page_ndcg"]), -_safe(summary[m]["context_precision"])),
    )
    best = ranked[0] if ranked else "—"

    base, _, judge_model = eval_settings._judge_endpoint(rag_settings)
    _, _, emb_model = eval_settings._embedding_endpoint(rag_settings)
    lines = [
        "# Retrieval mode comparison",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Questions: {len([r for r in per_question if r['mode'] == modes[0]]) if modes else 0}"
        f"  ·  k: {k}  ·  modes: {', '.join(modes)}",
        f"- Generator: `{rag_settings.generation_model}`  ·  Judge: `{judge_model}`"
        f"  ·  Embeddings: `{emb_model}`",
        f"- Endpoint: `{base}`",
        "",
        "## Summary (mean per mode)",
        "",
        "| metric | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for metric in SUMMARY_METRICS:
        cells = " | ".join(_fmt(summary[m][metric]) for m in modes)
        lines.append(f"| {metric} | {cells} |")
    lines += [
        "",
        f"**Recommended mode (by page nDCG, tie-break context precision): `{best}`**",
        "",
        "Deterministic IR metrics (`page_*`, `doc_*`) are the hard ranking; Ragas",
        "LLM-judge scores are directional. See `per_question.csv` for drill-down.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[eval] wrote results to {out_dir}/")


def _safe(v) -> float:
    return -1.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v


# -- main -----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Compare vector/pageindex/hybrid retrieval.")
    ap.add_argument("--n", type=int, default=None, help="max questions (default: all)")
    ap.add_argument("--modes", nargs="+", choices=ALL_MODES, default=None)
    ap.add_argument("--no-ragas", action="store_true", help="deterministic IR metrics only")
    ap.add_argument("--testset", default=None, help="testset jsonl (default from settings)")
    args = ap.parse_args()

    rag_settings, eval_settings = load_settings()
    k = eval_settings.retrieval_k(rag_settings)
    modes = args.modes or ALL_MODES
    rows = load_testset(args.testset or eval_settings.testset_path, args.n)
    if not rows:
        raise SystemExit("Test set is empty.")

    # Fail fast (and gently) if the endpoint is down, rather than retry-storming it.
    from rag.llm.factory import build_llm

    if not build_llm("generation", rag_settings).ping():
        raise SystemExit(
            f"LLM endpoint unreachable ({rag_settings.generation_base_url}). Is the GB10 on?"
        )

    print(f"[eval] {len(rows)} questions × {len(modes)} modes (k={k})")
    pipeline = RAGPipeline(rag_settings)
    per_question: list[dict] = []
    ragas_inputs: dict[str, list[dict | None]] = {m: [] for m in modes}

    for mode in modes:
        for i, row in enumerate(rows):
            q = row["question"]
            gold_doc = row.get("gold_doc_id")
            gold_pages = set(row.get("gold_pages") or [])
            rec: dict = {
                "mode": mode,
                "idx": i,
                "question": q,
                "gold_doc_id": gold_doc or "",
                "gold_pages": ",".join(str(p) for p in sorted(gold_pages)),
                "error": "",
            }
            try:
                t0 = time.perf_counter()
                ans = pipeline.ask(q, mode=mode, top_k=k)
                rec["latency_s"] = round(time.perf_counter() - t0, 3)
            except Exception as e:  # keep the run going; record the failure
                rec["error"] = repr(e)
                per_question.append(rec)
                ragas_inputs[mode].append(None)
                print(f"  [{mode} #{i}] ERROR: {e!r}")
                continue

            retrieved = [(c.doc_id, parse_locator(c.locator)) for c in ans.citations]
            rec.update(compute_ir(retrieved, gold_doc, gold_pages, k))
            rec["n_citations"] = len(ans.citations)
            rec["retrieved_docs"] = "; ".join(dict.fromkeys(c.doc_id for c in ans.citations))
            rec["answer"] = ans.text
            per_question.append(rec)
            ragas_inputs[mode].append(
                {
                    "question": q,
                    "response": ans.text,
                    "contexts": [c.text for c in ans.citations],
                    "reference": row.get("reference_answer", ""),
                }
            )
            print(f"  [{mode} #{i}] {rec['latency_s']}s  page_ndcg={_fmt(rec.get('page_ndcg'))}")

    if not args.no_ragas:
        for mode in modes:
            metas = [m for m in ragas_inputs[mode] if m is not None]
            if not metas:
                continue
            print(f"[eval] Ragas scoring {mode} ({len(metas)} questions)…")
            try:
                scores = run_ragas(metas, eval_settings, rag_settings)
            except Exception as e:
                print(f"  [warn] Ragas failed for {mode}: {e!r}")
                continue
            mode_rows = [r for r in per_question if r["mode"] == mode]
            score_iter = iter(scores)
            for rec, meta in zip(mode_rows, ragas_inputs[mode]):
                if meta is None:
                    continue
                rec.update(next(score_iter, {}))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_outputs(
        per_question, modes, k, eval_settings, rag_settings, Path(eval_settings.output_dir) / ts
    )


if __name__ == "__main__":
    main()
