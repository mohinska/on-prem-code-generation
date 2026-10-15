"""
onprem_codegen/eval/metrics.py — pass@k, compile rate and status breakdown for one group of samples.

A group is one (benchmark, task_mode, mode, decoding): every task in it has the same number of samples.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    prob_all_fail = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_fail *= 1.0 - k / i
    return 1.0 - prob_all_fail


def summarize(rows: list[dict], ks: list[int], exclude_tasks: set[str] = frozenset()) -> dict:
    """rows: executed samples with task_id, passed, compiled, status, finish_reason.
    pass@k is reported only when every task has at least k samples."""
    rows = [r for r in rows if r["task_id"] not in exclude_tasks]
    per_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_task[r["task_id"]].append(r)
    if not per_task:
        return {"n_tasks": 0, "n_samples": 0}
    min_n = min(len(v) for v in per_task.values())
    out = {"n_tasks": len(per_task), "n_samples": len(rows), "samples_per_task": min_n}
    for k in ks:
        if min_n >= k:
            scores = [pass_at_k(len(v), sum(r["passed"] for r in v), k) for v in per_task.values()]
            out[f"pass@{k}"] = sum(scores) / len(scores)
    out["compile_rate"] = sum(r["compiled"] for r in rows) / len(rows)
    out["no_program_rate"] = sum(r["status"] == "no_program" for r in rows) / len(rows)
    out["truncation_rate"] = sum(r.get("finish_reason") == "length" for r in rows) / len(rows)
    out["status"] = dict(Counter(r["status"] for r in rows))
    return out
