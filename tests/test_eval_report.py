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
