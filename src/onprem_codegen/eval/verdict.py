"""
onprem_codegen/eval/verdict.py: the sft-v1 verdict by the criteria fixed in docs/sft-v1-preregistration.md.

    python -m onprem_codegen.eval.verdict \\
        --base outputs/eval/baseline-v2/results.jsonl --base-frozen outputs/eval/baseline/results.jsonl \\
        --adapted outputs/eval/sft-v1/results.jsonl --seed2 outputs/eval/sft-v1-seed2/results.jsonl \\
        --heldout outputs/eval/heldout --out docs/sft-v1-verdict.md          # also writes verdict.json next to it

Baseline: baseline v2 groups (COBOLEval, CobolCodeBench completion: prompt fix) and the frozen baseline for the rest
(HumanEval-Python, CobolCodeBench instruct: prompts unchanged). Every criterion is read on the sampled decoding unless
it names greedy; the preregistration does not name the decoding of the guards and of the held-out test, so this rule
is fixed here before any sft-v1 result exists. Held-out: report_<model>.json from eval.heldout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onprem_codegen.eval.metrics import delta_ci, summarize, task_scores
from onprem_codegen.utils import read_jsonl

COBOL_S, COBOL_G = "coboleval/completion/non_thinking/sampled", "coboleval/completion/non_thinking/greedy"
PY_S = "humaneval_python/completion/non_thinking/sampled"
CCB_S = "cobolcodebench/completion/non_thinking/sampled"
PY_MAX_DROP, CCB_MAX_TRUNC_RISE = 0.02, 0.10


def group(r: dict) -> str:
    return "/".join((r["benchmark"], r["task_mode"], r["mode"], r["decoding"]))


def merge_baseline(v2: list[dict], frozen: list[dict]) -> list[dict]:
    """baseline v2 rows for the groups it has; frozen baseline rows for every other group."""
    have = {group(r) for r in v2}
    return v2 + [r for r in frozen if group(r) not in have]


def _by_group(rows):
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(group(r), []).append(r)
    return out


def _cmp(base, adapted, g, k=1):
    """Point values and the paired 95% CI of adapted - base for pass@k in group g."""
    b, a = base[g], adapted[g]
    sb, sa = summarize(b, [k]), summarize(a, [k])
    lo, hi = delta_ci(task_scores(b, k), task_scores(a, k))
    return {"base": sb[f"pass@{k}"], "adapted": sa[f"pass@{k}"], "delta": sa[f"pass@{k}"] - sb[f"pass@{k}"],
            "ci": [lo, hi], "compile": [sb["compile_rate"], sa["compile_rate"]],
            "trunc": [sb["truncation_rate"], sa["truncation_rate"]]}


def _main(base, adapted) -> tuple[bool, str]:
    m = _cmp(base, adapted, COBOL_S)
    ok = m["delta"] > 0 and m["ci"][0] > 0 and m["compile"][1] > m["compile"][0]
    return ok, (f"pass@1 {m['base']:.3f} → {m['adapted']:.3f}, Δ {m['delta']:+.3f} [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]; "
                f"compile {m['compile'][0]:.3f} → {m['compile'][1]:.3f}")


def _works(held: dict | None) -> tuple[bool, str]:
    if not held:
        return False, "no held-out report"
    d = held["delta"]["sampled"]["pass@1"]
    return d["mean"] > 0 and d["ci"][0] > 0, f"Δ pass@1 {d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"


def judge(base_rows, adapted_rows, held=None, seed2_rows=None, held2=None) -> dict:
    base, adapted = _by_group(base_rows), _by_group(adapted_rows)
    crit = []

    def add(name, title, met, detail):
        crit.append({"name": name, "title": title, "met": bool(met), "detail": detail})

    add("main", "Основний: COBOLEval sampled pass@1 вищий (нижня межа CI > 0) і вищий compile rate", *_main(base, adapted))
    p10 = _cmp(base, adapted, COBOL_S, 10)
    g1 = _cmp(base, adapted, COBOL_G)
    add("extra", "Додатковий: COBOLEval pass@10 вищий (нижня межа > 0), greedy pass@1 не нижчий",
        p10["delta"] > 0 and p10["ci"][0] > 0 and g1["delta"] >= 0,
        f"pass@10 Δ {p10['delta']:+.3f} [{p10['ci'][0]:+.3f}, {p10['ci'][1]:+.3f}]; greedy pass@1 Δ {g1['delta']:+.3f}")
    py = _cmp(base, adapted, PY_S)
    add("python_guard", f"Guard Python: HumanEval-Python sampled pass@1 падає не більше ніж на {PY_MAX_DROP}",
        -py["delta"] <= PY_MAX_DROP, f"pass@1 {py['base']:.3f} → {py['adapted']:.3f} (Δ {py['delta']:+.3f})")
    ccb = _cmp(base, adapted, CCB_S)
    rise = ccb["trunc"][1] - ccb["trunc"][0]
    add("ccb_guard", "Guard CobolCodeBench completion: pass@1 не нижчий за CI, обрізання не більше ніж +10 пунктів",
        ccb["ci"][1] >= 0 and rise <= CCB_MAX_TRUNC_RISE,
        f"Δ pass@1 {ccb['delta']:+.3f} [{ccb['ci'][0]:+.3f}, {ccb['ci'][1]:+.3f}]; обрізання "
        f"{ccb['trunc'][0]:.3f} → {ccb['trunc'][1]:.3f} ({rise:+.3f})")
    add("training_works", "«Навчання працює»: утримані задачі, pass@1 вищий за базу (нижня межа > 0)", *_works(held))

    by = {c["name"]: c for c in crit}
    main, py_ok, works = by["main"]["met"], by["python_guard"]["met"], by["training_works"]["met"]
    if not works:
        v = "невдача"
    elif main and py_ok:
        v = "успіх"
    elif main:
        v = "успіх без guard Python"   # not one of the three preregistered verdicts: slice C (Python) applies
    else:
        v = "частковий"

    seed2 = None
    if seed2_rows is not None:
        s_main = _main(base, _by_group(seed2_rows))
        s_works = _works(held2)
        seed2 = {"main": {"met": s_main[0], "detail": s_main[1]}, "training_works": {"met": s_works[0], "detail": s_works[1]}}
    unstable = bool(seed2) and (seed2["main"]["met"] != main or seed2["training_works"]["met"] != works)
    return {"verdict": v, "criteria": crit, "by_name": by, "seed2": seed2, "unstable": unstable}


def render(v: dict) -> str:
    lines = [f"**Вердикт: {v['verdict']}**" + (" — **нестабільний результат** (seed 42 і seed 43 розходяться)"
                                              if v["unstable"] else ""), "",
             "| Критерій | Виконано | Числа |", "|---|---|---|"]
    lines += [f"| {c['title']} | {'так' if c['met'] else 'ні'} | {c['detail']} |" for c in v["criteria"]]
    if v["seed2"]:
        lines += ["", "| Seed 43 (розкид) | Виконано | Числа |", "|---|---|---|"]
        lines += [f"| {n} | {'так' if c['met'] else 'ні'} | {c['detail']} |" for n, c in v["seed2"].items()]
    lines += ["", "Критерії: `docs/sft-v1-preregistration.md`; гарди й утримані задачі читаються на sampled."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="sft-v1 verdict by the preregistered criteria.")
    ap.add_argument("--base", type=Path, required=True, help="baseline v2 results.jsonl")
    ap.add_argument("--base-frozen", type=Path, required=True, help="frozen baseline results.jsonl (HumanEval, CCB instruct)")
    ap.add_argument("--adapted", type=Path, required=True, help="sft-v1 results.jsonl")
    ap.add_argument("--seed2", type=Path, help="sft-v1-seed2 results.jsonl")
    ap.add_argument("--heldout", type=Path, default=Path("outputs/eval/heldout"), help="dir with report_<model>.json")
    ap.add_argument("--out", type=Path, default=Path("outputs/eval/sft-v1-verdict.md"))
    a = ap.parse_args(argv)
    load = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    v = judge(merge_baseline(read_jsonl(a.base), read_jsonl(a.base_frozen)), read_jsonl(a.adapted),
              load(a.heldout / "report_cobol.json"), read_jsonl(a.seed2) if a.seed2 else None,
              load(a.heldout / "report_cobol-seed2.json"))
    md = render(v)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    a.out.with_name("verdict.json").write_text(json.dumps({k: v[k] for k in ("verdict", "criteria", "seed2", "unstable")},
                                                          indent=1, ensure_ascii=False), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
