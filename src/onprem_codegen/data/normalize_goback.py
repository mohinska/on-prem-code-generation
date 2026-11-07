"""sft-dataset:v0 -> sft-dataset:v1-lite: no held-out tasks, STOP RUN -> GOBACK, every program run again with its tests.

    python -m onprem_codegen.data.normalize_goback [--dataset sft-dataset:decontaminated | --input sft.jsonl]
        [--task-pool PATH] [--heldout PATH] [--out DIR] [--workers N] [--no-wandb]

Needs the eval-sandbox image (Docker), like rejection sampling. GOBACK ends a main program like STOP RUN but also
returns from a subprogram, which is how COBOLEval calls the solution. Programs whose rewritten version fails its
tests are dropped. Output: DIR/sft.jsonl, DIR/stats.json, W&B artifact sft-dataset (aliases v1-lite, and
decontaminated-v1 if the benchmark check finds no hit; the `decontaminated` alias of v0 is not touched).
"""
import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from onprem_codegen.eval.execute import extract_program
from onprem_codegen.utils import git_commit, read_jsonl

ARTIFACT = "sft-dataset:decontaminated"
HELDOUT = Path("data/splits/heldout_task_ids.json")
OUT = Path("data/raw/sft-dataset-v1-lite")
# `STOP RUN WITH ERROR STATUS` / `RETURNING` have no plain GOBACK equivalent here: left as they are.
STOP_RUN = re.compile(r"\bSTOP\s+RUN\b(?!\s+(?:WITH|RETURNING)\b)", re.I)
GOBACK = re.compile(r"\bGOBACK\b", re.I)
LITERAL = re.compile(r"\"[^\"]*\"|'[^']*'")


def stop_run_to_goback(program: str) -> tuple[str, int]:
    """Replaces the STOP RUN statements outside comment lines and string literals; returns (program, replaced)."""
    out, total = [], 0
    for line in program.split("\n"):
        if len(line) > 6 and line[6] in "*/":      # fixed-format comment line
            out.append(line)
            continue
        parts, pos = [], 0
        for m in LITERAL.finditer(line):
            parts.append(line[pos:m.start()])
            parts.append(m.group())                 # literals are kept as they are
            pos = m.end()
        parts.append(line[pos:])
        for i in range(0, len(parts), 2):           # even parts are code
            parts[i], n = STOP_RUN.subn("GOBACK", parts[i])
            total += n
        out.append("".join(parts))
    return "\n".join(out), total


def build_v1_lite(rows: list[dict], heldout: set[str], passes: Callable[[dict, str], bool],
                  workers: int = 1) -> tuple[list[dict], dict]:
    """Held-out tasks out, STOP RUN -> GOBACK (programs without any GOBACK are dropped), then `passes(row, program)` must hold."""
    stats = {"input": len(rows), "heldout": 0, "stop_run_replaced": 0, "no_goback": 0, "failed_recheck": 0, "kept": 0}
    todo = []
    for r in rows:
        if r["task_id"] in heldout:
            stats["heldout"] += 1
            continue
        program, n = stop_run_to_goback(extract_program(r["completion"]))
        stats["stop_run_replaced"] += n > 0
        if not GOBACK.search(program):             # no terminator at all: not the ending we want to teach
            stats["no_goback"] += 1
            continue
        completion = f"```cobol\n{program.strip(chr(10))}\n```"
        todo.append((dict(r, completion=completion, content_sha256=hashlib.sha256(completion.encode()).hexdigest()), program))
    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(lambda t: passes(*t), todo))
    kept = [r for (r, _), good in zip(todo, ok) if good]
    stats["failed_recheck"] = len(todo) - len(kept)
    stats["kept"] = len(kept)
    return kept, stats


def main(argv: list[str] | None = None) -> None:
    import yaml
    from onprem_codegen.data.decontaminate import run as decontaminate
    from onprem_codegen.data.rejection_sampling import all_tests, passes_ignoring_spacing

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=ARTIFACT, help="W&B sft-dataset version to start from")
    ap.add_argument("--input", help="local sft.jsonl instead of the W&B artifact")
    ap.add_argument("--task-pool", help="local task_pool.jsonl instead of the W&B artifact task-pool:v1")
    ap.add_argument("--heldout", default=str(HELDOUT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rs_cfg = yaml.safe_load(Path("configs/data/rejection_sampling.yaml").read_text())
    timeout = rs_cfg["execute"]["timeout_s"]

    run = None
    if not a.no_wandb:
        from onprem_codegen.utils import init_run
        run = init_run("sft-v1-lite", {"dataset": a.dataset, "heldout": a.heldout, "workers": a.workers})
    src = Path(a.input) if a.input else Path(run.use_artifact(a.dataset).download()) / "sft.jsonl"
    pool = Path(a.task_pool) if a.task_pool else Path(run.use_artifact(rs_cfg["task_pool"]).download()) / "task_pool.jsonl"
    tasks = {t["task_id"]: t for t in read_jsonl(pool)}
    heldout = set(json.loads(Path(a.heldout).read_text())["task_ids"])

    all_rows = read_jsonl(src)
    total = sum(r["task_id"] not in heldout for r in all_rows)   # upper bound: rows without GOBACK are skipped
    done, failed, lock, t0 = [0], [0], threading.Lock(), time.time()

    def passes(row: dict, program: str) -> bool:   # generated rows ran on every sample of the statement, human rows on its test
        tests = all_tests(tasks[row["task_id"]]) if row["source"] == "rejection_sampling" else tasks[row["task_id"]]["tests"]
        ok = passes_ignoring_spacing(program, tests, timeout)
        with lock:
            done[0] += 1
            failed[0] += not ok
            if done[0] % 20 == 0 or done[0] == total:
                eta = (time.time() - t0) / done[0] * (total - done[0]) / 60
                print(f"progress {done[0]}/{total} run, {failed[0]} failed, ~{eta:.0f} min left", flush=True)
        return ok

    rows, stats = build_v1_lite(all_rows, heldout, passes, a.workers)
    (out / "sft.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    stats["goback_share"] = round(sum("GOBACK" in r["completion"] for r in rows) / max(len(rows), 1), 4)
    stats["tasks"] = len({r["task_id"] for r in rows})
    stats["leaked_heldout"] = sum(r["task_id"] in heldout for r in rows)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)

    cfg = yaml.safe_load(Path("configs/data/decontam.yaml").read_text())
    rc = decontaminate(out / "sft.jsonl", out / "decontam_report.json", cfg)
    if run:
        import wandb
        art = wandb.Artifact("sft-dataset", type="dataset", metadata={**stats, "git_commit": git_commit(), "decontam_hits": rc})
        art.add_file(str(out / "sft.jsonl"))
        art.add_file(str(out / "stats.json"))
        run.log_artifact(art, aliases=["v1-lite"] + (["decontaminated-v1"] if rc == 0 else []))
        run.finish()


if __name__ == "__main__":
    main()
