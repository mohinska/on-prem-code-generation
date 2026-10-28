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

METRICS = ["pass@1", "pass@10", "compile_rate"]


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def _delta(a, b) -> str:
    return f"{b - a:+.3f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"


def render(base: dict, adapted: dict, base_name: str = "base", adapted_name: str = "adapted") -> str:
    header = ["group"]
    for m in METRICS:
        header += [f"{m} base", f"{m} adapted", "Δ"]
    lines = [f"# {adapted_name} vs {base_name}", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for group in sorted(base.keys() & adapted.keys()):
        row = [group]
        for m in METRICS:
            a, b = base[group].get(m), adapted[group].get(m)
            row += [_fmt(a), _fmt(b), _delta(a, b)]
        lines.append("| " + " | ".join(row) + " |")
    for name, only in ((base_name, base.keys() - adapted.keys()), (adapted_name, adapted.keys() - base.keys())):
        if only:
            lines += ["", f"Only in {name}: " + ", ".join(f"`{g}`" for g in sorted(only))]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Base vs adapted markdown table from two metrics.json files.")
    ap.add_argument("base", type=Path)
    ap.add_argument("adapted", type=Path)
    ap.add_argument("--out", type=Path, help="markdown file (default: print)")
    a = ap.parse_args()
    md = render(json.loads(a.base.read_text(encoding="utf-8")), json.loads(a.adapted.read_text(encoding="utf-8")),
                a.base.parent.name, a.adapted.parent.name)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
