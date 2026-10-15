import json
import subprocess
import sys
from pathlib import Path

import yaml

from onprem_codegen.data import rejection_sampling as rs
from onprem_codegen.prompt import build_prompt

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/data/rejection_sampling.yaml").read_text())
TASK = {"task_id": "codenet/p1", "prompt": "Print A+B.", "tests": [{"input": "3 5\n", "output": "8\n"}],
        "meta": {"problem_id": "p1"}}
PROG = "       PROCEDURE DIVISION.\n           DISPLAY \"OK\".\n"


def _sample(i, status="ok", program=PROG, hardcoded=False):
    return {"task_id": "codenet/p1", "sample_idx": i, "prompt": build_prompt("Print A+B."), "program": program,
            "status": status, "hardcoded": hardcoded}


def test_real_run_refuses_stub_executor():
    p = subprocess.run([sys.executable, "-m", "onprem_codegen.data.rejection_sampling", "--no-wandb",
                        "--task-pool", str(ROOT / "tests/fixtures/task_pool_tiny.jsonl")],
                       cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "EXECUTE_MODE": "stub"})
    assert p.returncode != 0 and "EXECUTE_MODE=real" in p.stderr


def test_hardcoded_detection():
    assert rs.is_hardcoded('           DISPLAY "8".', TASK["tests"])
    assert rs.is_hardcoded("           DISPLAY 8.", TASK["tests"])
    assert not rs.is_hardcoded("           DISPLAY WS-SUM.", TASK["tests"])
    assert not rs.is_hardcoded('           DISPLAY "Yes".', [{"input": "", "output": "Yes\n"}])


def test_build_sft_keeps_only_passing_distinct_capped():
    samples = [_sample(0), _sample(1),  # same program twice
               _sample(2, "test_fail", PROG + "*"), _sample(3, hardcoded=True, program=PROG + "**")]
    samples += [_sample(10 + i, program=PROG + f"      * v{i}\n") for i in range(6)]
    rows, dropped = rs.build_sft(samples, [], CFG)
    assert len(rows) == CFG["keep"]["max_per_task"]
    assert dropped == {"duplicate": 1, "hardcoded": 1, "over_cap": 3}
    r = rows[0]
    assert set(r) == {"prompt", "completion", "source", "task_id", "content_sha256", "meta"}
    assert r["completion"].startswith("```cobol\n       PROCEDURE") and r["completion"].endswith("\n```")
    assert r["source"] == "rejection_sampling"
    assert r["meta"] == {"generator_model": CFG["generator_model"], "submission_id": None}


def test_dry_run_end_to_end(tmp_path):
    out = tmp_path / "sft"
    subprocess.run([sys.executable, "-m", "onprem_codegen.data.rejection_sampling", "--dry-run", "--no-wandb",
                    "--task-pool", str(ROOT / "tests/fixtures/task_pool_tiny.jsonl"), "--out", str(out)],
                   cwd=ROOT, check=True, env={"PYTHONPATH": str(ROOT / "src"), "EXECUTE_MODE": "stub"})
    stats = json.loads((out / "stats.json").read_text())
    assert stats["tasks"] == 3 and stats["samples"] == 3 * CFG["sampling"]["n"]
    assert stats["by_status"]["no_program"] > 0
    rows = [json.loads(l) for l in open(out / "sft.jsonl")]
    assert rows and all(r["source"] == "rejection_sampling" for r in rows)
    # restart: nothing re-sampled, same output
    subprocess.run([sys.executable, "-m", "onprem_codegen.data.rejection_sampling", "--dry-run", "--no-wandb",
                    "--task-pool", str(ROOT / "tests/fixtures/task_pool_tiny.jsonl"), "--out", str(out)],
                   cwd=ROOT, check=True, env={"PYTHONPATH": str(ROOT / "src"), "EXECUTE_MODE": "stub"})
    assert json.loads((out / "stats.json").read_text()) == stats
