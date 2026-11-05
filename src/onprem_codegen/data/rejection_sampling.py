"""Step 3: task-pool -> n samples per task on A's vLLM server -> run_program -> sft-dataset (status ok only).

    python -m onprem_codegen.data.rejection_sampling [--task-pool PATH] [--clean-corpus DIR] [--limit N] [--no-wandb]

Runs where both the server (SSH tunnel to localhost:8000) and the eval sandbox are reachable; Vast has no Docker.
Needs the image from `docker compose build eval-sandbox` (or EXECUTE_BACKEND=local on a disposable host).
Output: data/raw/sft-dataset/{sft.jsonl (contracts.md: sft-dataset), samples.jsonl (every generation with
its status, input for DPO pairs later), stats.json}. Restartable: tasks already in samples.jsonl are skipped.
"""
import argparse
import collections
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from onprem_codegen.eval.execute import execute, extract_program
from onprem_codegen.eval.generate import VllmClient
from onprem_codegen.prompt import build_messages, build_prompt
from onprem_codegen.utils import git_commit, read_jsonl

CFG = Path("configs/data/rejection_sampling.yaml")
OUT = Path("data/raw/sft-dataset")


def run_program(source: str, tests: list[dict], timeout_s: float) -> dict:
    """Compile `source` and run it once per {"input", "output"} test (stdin -> stdout); tests=[] only compiles."""
    r = execute(source, [{"input": t["input"], "expected": t["output"]} for t in tests], timeout_s=timeout_s)
    return {"status": r.status.value, "stderr": r.compile_stderr or r.run_stderr, "duration_s": r.duration_s}


def is_hardcoded(program: str, tests: list[dict]) -> bool:
    """Every test's (numeric) expected output is DISPLAYed as a literal: likely a memorized sample answer."""
    if not tests:
        return False
    for t in tests:
        first = t["output"].strip().split("\n")[0].strip()
        if not any(c.isdigit() for c in first):
            return False
        if not re.search(r"DISPLAY\s+[\"']?" + re.escape(first) + r"[\"']?(\s|\.|$)", program, re.I | re.M):
            return False
    return True


def sample_task(task: dict, client, cfg: dict) -> list[dict]:
    rows = []
    messages = build_messages(task["prompt"])
    if cfg.get("generation_hints"):  # only sent to the model; the saved prompt stays build_prompt, same as eval
        messages[0]["content"] += "\n\n" + cfg["generation_hints"]
    for i, g in enumerate(client.generate(messages, cfg["sampling"], False)):
        gen, finish = g["content"], g["finish_reason"]
        prog = extract_program(gen)
        res = (run_program(prog, task["tests"], cfg["execute"]["timeout_s"]) if prog
               else {"status": "no_program", "stderr": "", "duration_s": 0.0})
        rows.append({"task_id": task["task_id"], "sample_idx": i, "prompt": build_prompt(task["prompt"]),
                     "generation": gen, "finish_reason": finish, "program": prog, "status": res["status"],
                     "stderr": res["stderr"][:500], "duration_s": res["duration_s"],
                     "hardcoded": bool(prog) and is_hardcoded(prog, task["tests"])})
    return rows


def sft_row(prompt: str, program: str, source: str, task_id: str, generator: str | None, submission_id: str | None) -> dict:
    completion = f"```cobol\n{program.strip(chr(10))}\n```"
    return {"prompt": prompt, "completion": completion, "source": source, "task_id": task_id,
            "content_sha256": hashlib.sha256(completion.encode()).hexdigest(),
            "meta": {"generator_model": generator, "submission_id": submission_id}}


def build_sft(samples: list[dict], human: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """Passing, distinct programs; at most keep.max_per_task per (task, source)."""
    keep, seen, per = [], set(), collections.Counter()
    drop = collections.Counter()
    gen_rows = []
    for s in sorted(samples, key=lambda s: (s["task_id"], s["sample_idx"])):
        if s["status"] != "ok":
            continue
        if cfg["keep"]["drop_hardcoded_output"] and s["hardcoded"]:
            drop["hardcoded"] += 1
            continue
        gen_rows.append(sft_row(s["prompt"], s["program"], "rejection_sampling", s["task_id"], cfg["generator_model"], None))
    for r in gen_rows + human:
        if r["content_sha256"] in seen:
            drop["duplicate"] += 1
            continue
        if per[r["task_id"], r["source"]] >= cfg["keep"]["max_per_task"]:
            drop["over_cap"] += 1
            continue
        seen.add(r["content_sha256"])
        per[r["task_id"], r["source"]] += 1
        keep.append(r)
    return keep, dict(drop)


def human_rows(clean_dir: Path, tasks: dict, cfg: dict) -> list[dict]:
    """Accepted CodeNet submissions that survived clean-corpus and pass the tests with our cobc."""
    rows = []
    by_pid = {t["meta"]["problem_id"]: t for t in tasks.values()}
    for line in open(clean_dir / "corpus.jsonl"):
        d = json.loads(line)
        task = by_pid.get(d["meta"].get("problem_id")) if d["source"] == "codenet_cobol" else None
        if task and run_program(d["content"], task["tests"], cfg["execute"]["timeout_s"])["status"] == "ok":
            rows.append(sft_row(build_prompt(task["prompt"]), d["content"], "codenet_cobol", task["task_id"],
                                None, d["meta"]["submission_id"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-pool", help="local task_pool.jsonl instead of the W&B artifact")
    ap.add_argument("--clean-corpus", help="local clean-corpus dir instead of the W&B artifact")
    ap.add_argument("--limit", type=int, help="only the first N tasks")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    run = None
    if not a.no_wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "onprem-codegen"), job_type="rejection-sampling",
                         config={**cfg, "git_commit": git_commit(),
                                 "vast_instance": os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL")})
    pool = Path(a.task_pool) if a.task_pool else Path(run.use_artifact(cfg["task_pool"]).download()) / "task_pool.jsonl"
    tasks = {t["task_id"]: t for t in read_jsonl(pool)[:a.limit]}

    srv = cfg["server"]
    client = VllmClient(srv["base_url"], srv["model"], srv["request_timeout_s"])
    samples_path = out / "samples.jsonl"
    done = {s["task_id"] for s in read_jsonl(samples_path)}
    todo = [t for tid, t in tasks.items() if tid not in done]
    print(f"{len(tasks)} tasks, {len(done)} already sampled, {len(todo)} to go")

    with open(samples_path, "a") as f, ThreadPoolExecutor(cfg["concurrency"]) as pool_ex:
        futures = [pool_ex.submit(sample_task, t, client, cfg) for t in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            f.write("".join(json.dumps(s) + "\n" for s in fut.result()))  # one write per task
            f.flush()
            if k % 50 == 0:
                print(f"{k}/{len(todo)} tasks")

    samples = list({(s["task_id"], s["sample_idx"]): s for s in read_jsonl(samples_path) if s["task_id"] in tasks}.values())
    human = []
    if a.clean_corpus or (run and cfg["clean_corpus"]):
        clean = Path(a.clean_corpus) if a.clean_corpus else Path(run.use_artifact(cfg["clean_corpus"]).download())
        human = human_rows(clean, tasks, cfg)
    rows, dropped = build_sft(samples, human, cfg)

    with open(out / "sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    solved = {s["task_id"] for s in samples if s["status"] == "ok"}
    stats = {"tasks": len(tasks), "samples": len(samples), "tasks_solved": len(solved),
             "solve_rate": round(len(solved) / max(len(tasks), 1), 4),
             "sample_pass_rate": round(sum(s["status"] == "ok" for s in samples) / max(len(samples), 1), 4),
             "sft_rows": len(rows), "sft_tasks": len({r["task_id"] for r in rows}),
             "by_status": dict(collections.Counter(s["status"] for s in samples)),
             "by_source": dict(collections.Counter(r["source"] for r in rows)), "dropped": dropped,
             "truncated": sum(s["finish_reason"] == "length" for s in samples),
             "generator_model": cfg["generator_model"]}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    if run:
        import wandb
        run.summary.update({k: v for k, v in stats.items() if not isinstance(v, dict)})
        cols = ["task_id", "sample_idx", "prompt", "generation", "program", "status", "stderr", "duration_s"]
        run.log({"samples": wandb.Table(columns=cols, data=[[s[c] for c in cols] for s in samples[:1000]])})
        art = wandb.Artifact("sft-dataset", type="dataset", metadata=stats)
        for name in ("sft.jsonl", "samples.jsonl", "stats.json"):
            art.add_file(str(out / name))
        run.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    main()
