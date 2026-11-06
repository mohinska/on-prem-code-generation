"""Step 3: task-pool -> n samples per task on A's vLLM server -> run_program -> sft-dataset (status ok only).

    python -m onprem_codegen.data.rejection_sampling [--task-pool PATH] [--clean-corpus DIR] [--limit N] [--second-pass] [--no-wandb]

Runs where both the server (SSH tunnel to localhost:8000) and the eval sandbox are reachable; Vast has no Docker.
Needs the image from `docker compose build eval-sandbox` (or EXECUTE_BACKEND=local on a disposable host).
Output: data/raw/sft-dataset/{sft.jsonl (contracts.md: sft-dataset), samples.jsonl (every generation with
its status, input for DPO pairs later), stats.json}. Restartable: tasks already in samples.jsonl are skipped.
Generated programs run on every sample of the statement (all_tests); older passing ones are re-run on the next start.
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

from onprem_codegen.eval.execute import Status, execute, extract_program
from onprem_codegen.eval.generate import VllmClient
from onprem_codegen.prompt import build_messages, build_prompt
from onprem_codegen.utils import git_commit, read_jsonl

CFG = Path("configs/data/rejection_sampling.yaml")
OUT = Path("data/raw/sft-dataset")
SAMPLE = re.compile(r"Sample\s*Input\s*(\d*)\s*\n(.*?)\n\s*"
                    r"(?:Sample\s*Output(?:\s*for\s*Input)?|Output\s*for\s*(?:the\s*)?Sample\s*Input)\s*\1\s*\n+"
                    r"(.*?)(?=\n\s*\n|\n\s*Sample\s*Input|\Z)", re.S | re.I)


def run_program(source: str, tests: list[dict], timeout_s: float) -> dict:
    """Compile `source` and run it once per {"input", "output"} test (stdin -> stdout); tests=[] only compiles."""
    r = execute(source, [{"input": t["input"], "expected": t["output"]} for t in tests], timeout_s=timeout_s)
    return {"status": r.status.value, "stderr": r.compile_stderr or r.run_stderr, "duration_s": r.duration_s}


def all_tests(task: dict) -> list[dict]:
    """The task's test plus the other samples of its statement, if parsing reproduces that test (one test lets wrong programs through)."""
    parsed = [{"input": i.strip("\n") + "\n", "output": o.strip() + "\n"} for _, i, o in SAMPLE.findall(task["prompt"])]
    own = task["tests"][0]
    same = parsed and parsed[0]["input"].split() == own["input"].split() and parsed[0]["output"].split() == own["output"].split()
    return task["tests"] + parsed[1:] if same else task["tests"]


def passes_ignoring_spacing(source: str, tests: list[dict], timeout_s: float) -> bool:
    """Contest judges accept `  42` for 42, the sandbox doesn't: its wrong_output is checked again here, token by token."""
    for t in tests:
        r = execute(source, [{"input": t["input"], "expected": t["output"]}], timeout_s=timeout_s)
        if r.status != Status.OK and not (r.status == Status.WRONG_OUTPUT and r.run_stdout.split() == t["output"].split()):
            return False
    return True


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


def task_score(prompt: str) -> int | None:
    m = re.match(r"Score\s*:\s*(\d+)", prompt)
    return int(m.group(1)) if m else None


def worth_sampling(task: dict, max_score: int | None) -> bool:
    """AtCoder tasks up to max_score; harder ones and tasks without a score (AOJ) almost never yield a passing program."""
    if max_score is None:
        return True
    score = task_score(task["prompt"])
    return score is not None and score <= max_score


def select_todo(tasks: dict, samples: list[dict], cfg: dict, second_pass: bool) -> list[dict]:
    """Tasks without samples, or with second_pass the easy tasks whose n samples all failed (more than n: already done)."""
    seen = collections.Counter(s["task_id"] for s in samples)
    if not second_pass:
        return [t for tid, t in tasks.items() if not seen[tid] and worth_sampling(t, cfg.get("max_score"))]
    solved = {s["task_id"] for s in samples if s["status"] == "ok"}
    return [t for tid, t in tasks.items() if seen[tid] == cfg["sampling"]["n"] and tid not in solved
            and worth_sampling(t, cfg["second_pass_max_score"])]


def sample_task(task: dict, client, cfg: dict, start_idx: int = 0) -> list[dict]:
    rows = []
    messages = build_messages(task["prompt"])
    if cfg.get("generation_hints"):  # only sent to the model; the saved prompt stays build_prompt, same as eval
        messages[0]["content"] += "\n\n" + cfg["generation_hints"]
    tests = all_tests(task)
    for i, g in enumerate(client.generate(messages, cfg["sampling"], False), start_idx):
        gen, finish = g["content"], g["finish_reason"]
        prog = extract_program(gen)
        res = (run_program(prog, tests, cfg["execute"]["timeout_s"]) if prog
               else {"status": "no_program", "stderr": "", "duration_s": 0.0})
        rows.append({"task_id": task["task_id"], "sample_idx": i, "prompt": build_prompt(task["prompt"]),
                     "generation": gen, "finish_reason": finish, "program": prog, "status": res["status"],
                     "stderr": res["stderr"][:500], "duration_s": res["duration_s"], "n_tests": len(tests),
                     "hardcoded": bool(prog) and is_hardcoded(prog, task["tests"])})
    return rows


def recheck_ok(samples: list[dict], tasks: dict, cfg: dict) -> int:
    """Passing samples that ran on fewer tests than all_tests gives: run them again and update them. Returns how many ran."""
    todo = [(s, all_tests(tasks[s["task_id"]])) for s in samples if s["status"] == "ok" and s["task_id"] in tasks]
    todo = [(s, tests) for s, tests in todo if s.get("n_tests", 1) < len(tests)]
    with ThreadPoolExecutor(cfg["concurrency"]) as ex:
        results = ex.map(lambda st: run_program(st[0]["program"], st[1], cfg["execute"]["timeout_s"]), todo)
        for (s, tests), r in zip(todo, results):
            s.update(status=r["status"], stderr=r["stderr"][:500], duration_s=r["duration_s"], n_tests=len(tests))
    return len(todo)


def sft_row(prompt: str, program: str, source: str, task_id: str, generator: str | None, submission_id: str | None) -> dict:
    completion = f"```cobol\n{program.strip(chr(10))}\n```"
    return {"prompt": prompt, "completion": completion, "source": source, "task_id": task_id,
            "content_sha256": hashlib.sha256(completion.encode()).hexdigest(),
            "meta": {"generator_model": generator, "submission_id": submission_id}}


def build_sft(samples: list[dict], human: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """Passing, distinct programs; at most keep.max_per_task per (task, source). min_tests and max_score: generated rows only."""
    keep, seen, per = [], set(), collections.Counter()
    drop = collections.Counter()
    gen_rows = []
    for s in sorted(samples, key=lambda s: (s["task_id"], s["sample_idx"])):
        if s["status"] != "ok":
            continue
        if cfg["keep"]["drop_hardcoded_output"] and s["hardcoded"]:
            drop["hardcoded"] += 1
            continue
        if s.get("n_tests", 1) < cfg["keep"]["min_tests"]:
            drop["few_tests"] += 1
            continue
        if (task_score(s["prompt"]) or 0) > cfg["keep"]["max_score"]:
            drop["too_hard"] += 1
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
    """Accepted CodeNet submissions that survived clean-corpus: the judge checked the logic, here only that they work with our cobc."""
    by_pid = {t["meta"]["problem_id"]: t for t in tasks.values()}
    todo = []
    for line in open(clean_dir / "corpus.jsonl"):
        d = json.loads(line)
        task = by_pid.get(d["meta"].get("problem_id")) if d["source"] == "codenet_cobol" else None
        if task:
            todo.append((d, task))
    with ThreadPoolExecutor(cfg["concurrency"]) as ex:
        passed = ex.map(lambda dt: passes_ignoring_spacing(dt[0]["content"], dt[1]["tests"], cfg["execute"]["timeout_s"]), todo)
        return [sft_row(build_prompt(task["prompt"]), d["content"], "codenet_cobol", task["task_id"],
                        None, d["meta"]["submission_id"]) for (d, task), ok in zip(todo, passed) if ok]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-pool", help="local task_pool.jsonl instead of the W&B artifact")
    ap.add_argument("--clean-corpus", help="local clean-corpus dir instead of the W&B artifact")
    ap.add_argument("--limit", type=int, help="only the first N tasks")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--second-pass", action="store_true", help="n more samples for unsolved tasks up to second_pass_max_score")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    run = None
    if not a.no_wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "onprem-codegen"), job_type="rejection-sampling",
                         config={**cfg, "git_commit": git_commit()})
    pool = Path(a.task_pool) if a.task_pool else Path(run.use_artifact(cfg["task_pool"]).download()) / "task_pool.jsonl"
    tasks = {t["task_id"]: t for t in read_jsonl(pool)[:a.limit]}

    srv = cfg["server"]
    client = VllmClient(srv["base_url"], srv["model"], srv["request_timeout_s"])
    samples_path = out / "samples.jsonl"
    prior = read_jsonl(samples_path)
    todo = select_todo(tasks, prior, cfg, a.second_pass)
    print(f"{len(tasks)} tasks, {len({s['task_id'] for s in prior})} already sampled, {len(todo)} to go")
    start_idx = cfg["sampling"]["n"] if a.second_pass else 0

    with open(samples_path, "a") as f, ThreadPoolExecutor(cfg["concurrency"]) as pool_ex:
        futures = [pool_ex.submit(sample_task, t, client, cfg, start_idx) for t in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            f.write("".join(json.dumps(s) + "\n" for s in fut.result()))  # one write per task
            f.flush()
            if k % 50 == 0:
                print(f"{k}/{len(todo)} tasks")

    all_samples = read_jsonl(samples_path)
    if rechecked := recheck_ok(all_samples, tasks, cfg):
        print(f"re-ran {rechecked} passing samples on all statement samples")
        tmp = samples_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(s) + "\n" for s in all_samples))
        tmp.replace(samples_path)
    samples = list({(s["task_id"], s["sample_idx"]): s for s in all_samples if s["task_id"] in tasks}.values())
    human = []
    if a.clean_corpus or (run and cfg["clean_corpus"]):
        clean = Path(a.clean_corpus) if a.clean_corpus else Path(run.use_artifact(cfg["clean_corpus"]).download())
        human = human_rows(clean, tasks, cfg)
    rows, dropped = build_sft(samples, human, cfg)

    with open(out / "sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    solved = {s["task_id"] for s in samples if s["status"] == "ok"}
    stats = {"tasks": len(tasks), "tasks_sampled": len({s["task_id"] for s in samples}), "samples": len(samples),
             "tasks_solved": len(solved),
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
