"""Retrieval-mode evaluation harness (vector vs pageindex vs hybrid).

Additive to the core ``rag`` package — nothing here is imported by the running
service. Two layers:

* :mod:`eval.metrics` — deterministic IR metrics (stdlib only, unit-tested
  offline): the *hard* ranking of which mode retrieves the right pages.
* :mod:`eval.gen_testset` / :mod:`eval.run_eval` — Ragas-based test-set
  synthesis and LLM-judge RAG quality, run against the configured endpoints.

See ``EVAL_PLAN.md`` and ``eval/README.md``.
"""
