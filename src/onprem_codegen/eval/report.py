"""
onprem_codegen/eval/report.py — base vs adapted markdown table from two eval runs.

    python -m onprem_codegen.eval.report outputs/eval/baseline/metrics.json \
        outputs/eval/sft-v0/metrics.json --out docs/sft-v0.md

Inputs are the metrics.json files written by eval/run.py: {"<benchmark>/<task_mode>/<mode>/<decoding>": summary}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onprem_codegen.eval.metrics import boot_ci, delta_ci, task_scores

METRICS = ["pass@1", "pass@10", "compile_rate"]


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def _delta(a, b) -> str:
    return f"{b - a:+.3f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"


def render(base: dict, adapted: dict, base_name: str = "base", adapted_name: str = "adapted",
           ci: dict | None = None) -> str:
    """ci: {group: (lo, hi)} of the pass@1 difference; adds a column when given."""
    header = ["group"]
    for m in METRICS:
        header += [f"{m} base", f"{m} adapted", "Δ"]
    if ci is not None:
        header.append("Δ pass@1 95% CI")
    lines = [f"# {adapted_name} vs {base_name}", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for group in sorted(base.keys() & adapted.keys()):
        row = [group]
        for m in METRICS:
            a, b = base[group].get(m), adapted[group].get(m)
            row += [_fmt(a), _fmt(b), _delta(a, b)]
        if ci is not None:
            row.append(f"[{ci[group][0]:+.3f}, {ci[group][1]:+.3f}]" if group in ci else "-")
        lines.append("| " + " | ".join(row) + " |")
    for name, only in ((base_name, base.keys() - adapted.keys()), (adapted_name, adapted.keys() - base.keys())):
        if only:
            lines += ["", f"Only in {name}: " + ", ".join(f"`{g}`" for g in sorted(only))]
    return "\n".join(lines) + "\n"


def _read_rows(metrics_path: Path) -> list[dict]:
    """results.jsonl next to metrics.json (per-sample rows), [] if missing."""
    p = metrics_path.parent / "results.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.is_file() else []


def _by_group(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault("/".join((r["benchmark"], r["task_mode"], r["mode"], r["decoding"])), []).append(r)
    return out


def pass1_cis(base_rows: list[dict], adapted_rows: list[dict]) -> dict[str, tuple[float, float]]:
    """{group: 95% CI of the pass@1 difference} for the groups present in both result files."""
    b, a = _by_group(base_rows), _by_group(adapted_rows)
    return {g: delta_ci(task_scores(b[g], 1), task_scores(a[g], 1)) for g in sorted(b.keys() & a.keys())}


def render_ci(rows: list[dict], ks=(1, 10)) -> str:
    """One run: pass@k with a 95% bootstrap CI over tasks per group; '-' where a task has fewer than k samples."""
    lines = ["| group | tasks | " + " | ".join(f"pass@{k} [95% CI]" for k in ks) + " |", "|" + "---|" * (len(ks) + 2)]
    for g, grows in sorted(_by_group(rows).items()):
        n_min = min(sum(r["task_id"] == t for r in grows) for t in {r["task_id"] for r in grows})
        cells = []
        for k in ks:
            if n_min < k:
                cells.append("-")
                continue
            s = list(task_scores(grows, k).values())
            lo, hi = boot_ci(s)
            cells.append(f"{sum(s) / len(s):.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {g} | {len({r['task_id'] for r in grows})} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Base vs adapted markdown table from two metrics.json files.")
    ap.add_argument("base", type=Path, nargs="?")
    ap.add_argument("adapted", type=Path, nargs="?")
    ap.add_argument("--ci-of", type=Path, nargs="+", metavar="results.jsonl",
                    help="instead: one table with pass@k and 95%% CI per group of these results.jsonl files")
    ap.add_argument("--out", type=Path, help="markdown file (default: print)")
    a = ap.parse_args()
    if a.ci_of:
        md = render_ci([json.loads(l) for p in a.ci_of for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])
    elif a.base and a.adapted:
        base_rows, adapted_rows = _read_rows(a.base), _read_rows(a.adapted)
        ci = pass1_cis(base_rows, adapted_rows) if base_rows and adapted_rows else None   # needs results.jsonl of both
        md = render(json.loads(a.base.read_text(encoding="utf-8")), json.loads(a.adapted.read_text(encoding="utf-8")),
                    a.base.parent.name, a.adapted.parent.name, ci)
    else:
        ap.error("give base and adapted metrics.json, or --ci-of results.jsonl ...")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
