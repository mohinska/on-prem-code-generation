"""eval/report.py: base vs adapted markdown table."""
import json
import os
import subprocess
import sys
from pathlib import Path

from onprem_codegen.eval.report import render

ROOT = Path(__file__).resolve().parent.parent
G = "coboleval/completion/non_thinking/greedy"
BASE = {G: {"pass@1": 0.12, "compile_rate": 0.61}, "only/base/x/y": {"pass@1": 0.5}}
ADAPTED = {G: {"pass@1": 0.25, "compile_rate": 0.84}, "only/adapted/x/y": {"pass@1": 0.5}}


def test_row_with_delta_and_missing_metric():
    md = render(BASE, ADAPTED)
    # pass@10 is missing for greedy -> "-" in all three of its columns
    assert f"| {G} | 0.120 | 0.250 | +0.130 | - | - | - | 0.610 | 0.840 | +0.230 |" in md


def test_negative_delta_has_a_sign():
    md = render({G: {"pass@1": 0.3}}, {G: {"pass@1": 0.28}})
    assert "| 0.300 | 0.280 | -0.020 |" in md


def test_groups_in_one_file_only_are_listed():
    md = render(BASE, ADAPTED)
    assert "Only in base: `only/base/x/y`" in md
    assert "Only in adapted: `only/adapted/x/y`" in md


def test_cli_writes_the_file(tmp_path):
    (tmp_path / "baseline").mkdir()
    (tmp_path / "sft-v0").mkdir()
    base, adapted = tmp_path / "baseline/metrics.json", tmp_path / "sft-v0/metrics.json"
    base.write_text(json.dumps(BASE), encoding="utf-8")
    adapted.write_text(json.dumps(ADAPTED), encoding="utf-8")
    out = tmp_path / "reports/sft-v0.md"
    p = subprocess.run([sys.executable, "-m", "onprem_codegen.eval.report", str(base), str(adapted), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert p.returncode == 0, p.stderr
    md = out.read_text(encoding="utf-8")
    assert md.startswith("# sft-v0 vs baseline") and "+0.130" in md


# ---------- bootstrap CI of the pass@1 difference (paired by task) ----------
from onprem_codegen.eval.metrics import delta_ci, task_scores


def test_task_scores_are_pass_at_k_per_task():
    rows = [{"task_id": "a", "passed": True}, {"task_id": "a", "passed": False},
            {"task_id": "b", "passed": False}, {"task_id": "b", "passed": False}]
    assert task_scores(rows, 1) == {"a": 0.5, "b": 0.0}


def test_clear_gain_has_a_ci_above_zero():
    tasks = {f"t{i}": 0.0 for i in range(20)}
    lo, hi = delta_ci(tasks, {t: 1.0 for t in tasks})
    assert lo == hi == 1.0


def test_pure_noise_ci_contains_zero():
    base = {f"t{i}": float(i % 2) for i in range(40)}
    adapted = {t: 1.0 - v for t, v in base.items()}      # same mean, different tasks
    lo, hi = delta_ci(base, adapted)
    assert lo < 0 < hi


def test_ci_uses_only_tasks_in_both_runs_and_is_reproducible():
    base, adapted = {"a": 0.0, "b": 0.0, "only_base": 1.0}, {"a": 1.0, "b": 1.0, "only_adapted": 0.0}
    assert delta_ci(base, adapted) == delta_ci(base, adapted) == (1.0, 1.0)


def test_render_adds_the_ci_column_only_when_given():
    assert "CI" not in render(BASE, ADAPTED)
    md = render(BASE, ADAPTED, ci={G: (0.05, 0.21)})
    assert "Δ pass@1 95% CI" in md and f"| {G} |" in md and "[+0.050, +0.210]" in md


# ---------- CI of one run (the frozen baseline table) ----------
from onprem_codegen.eval.metrics import boot_ci
from onprem_codegen.eval.report import render_ci


def _rows(decoding, n_samples, passing):
    """40 tasks, `passing` of them solved by every sample, the rest by none."""
    return [{"benchmark": "coboleval", "task_mode": "completion", "mode": "non_thinking", "decoding": decoding,
             "task_id": f"t{i}", "passed": i < passing} for i in range(40) for _ in range(n_samples)]


def test_boot_ci_of_a_constant_is_that_constant():
    assert boot_ci([0.5] * 10) == (0.5, 0.5)


def test_render_ci_shows_mean_with_interval_and_dash_when_too_few_samples():
    md = render_ci(_rows("greedy", 1, 20) + _rows("sampled", 10, 4), ks=[1, 10])
    greedy = next(l for l in md.splitlines() if "/greedy |" in l)
    sampled = next(l for l in md.splitlines() if "/sampled |" in l)
    assert "| 40 | 0.500 [" in greedy and greedy.rstrip().endswith("| - |")      # n=1 -> no pass@10
    assert "| 40 | 0.100 [" in sampled and sampled.count("0.100 [") == 2          # pass@1 and pass@10


def test_cli_ci_of_prints_a_single_run_table(tmp_path):
    f = tmp_path / "results.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in _rows("greedy", 1, 20)), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "onprem_codegen.eval.report", "--ci-of", str(f)], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert p.returncode == 0, p.stderr
    assert "| coboleval/completion/non_thinking/greedy | 40 | 0.500 [" in p.stdout
