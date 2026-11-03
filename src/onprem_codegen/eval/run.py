"""
onprem_codegen/eval/run.py — eval: generate -> execute -> metrics -> W&B.

    # one place (laptop: SSH tunnel to vLLM on Vast, docker sandbox locally)
    python -m onprem_codegen.eval.run --config configs/eval/baseline.yaml

    # split, because Vast has no docker
    python -m onprem_codegen.eval.run --stage generate                       # on Vast: logs eval-generations
    python -m onprem_codegen.eval.run --stage execute \\
        --generations eval-generations:baseline                              # locally, in the sandbox

    # one mode at a time into the same run (non-thinking is fast, thinking is the long run)
    python -m onprem_codegen.eval.run --stage generate --modes non_thinking

Both stages append to jsonl files under <out_dir>/<run_name>/ and skip finished work, so a crashed or
interrupted run resumes where it stopped. Generations are raw model output; the program is extracted
and assembled at execution time (eval/generate.py), so a fix there needs no new generations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from onprem_codegen.eval import benchmarks
from onprem_codegen.eval.benchmarks import Task
from onprem_codegen.eval.execute import execute
from onprem_codegen.eval.generate import VllmClient, messages_for, program_from_answer, split_reasoning
from onprem_codegen.eval.metrics import summarize
from onprem_codegen.utils import git_commit, read_jsonl

CONFIG = Path("configs/eval/baseline.yaml")
TABLE_TEXT = 8000          # chars per text cell in the W&B Table; full text is in the artifacts
TABLE_REASONING = 2000


# ---------- helpers ----------

def protocol(cfg: dict | None = None) -> dict:
    """evaluation_protocol from the manifest; cfg["protocol_overrides"] (e.g. a cheaper smoke run)
    replaces keys per mode: {thinking: {sampled: {n: 10}}}."""
    proto = yaml.safe_load(benchmarks.MANIFEST.read_text(encoding="utf-8"))["evaluation_protocol"]
    for mode, over in ((cfg or {}).get("protocol_overrides") or {}).items():
        for key, val in over.items():
            m = proto["modes"][mode]
            m[key] = {**m[key], **val} if isinstance(val, dict) else val
    return proto


def sampling_plan(proto: dict, mode: str, decoding: str, seed: int) -> list[dict] | None:
    """One request body per run: greedy repeats `runs` times (n=1), sampled is one request of n."""
    m = proto["modes"][mode]
    if decoding not in m:
        return None
    base = {**m[decoding], "max_tokens": m["max_new_tokens"]}
    runs = proto["runs"] if decoding == "greedy" else 1
    return [{**base, "seed": seed + r} for r in range(runs)]


def load_tasks(cfg: dict, limit: int | None) -> dict[tuple[str, str, str], Task]:
    """{(benchmark, task_mode, task_id): Task}"""
    out = {}
    for name, task_modes in cfg["benchmarks"].items():
        for tm in task_modes:
            for t in benchmarks.load(name, tm)[:limit]:
                out[(t.benchmark, t.mode, t.task_id)] = t
    return out


def _unit(row: dict) -> tuple:
    return row["benchmark"], row["task_mode"], row["mode"], row["decoding"], row["task_id"], row["run"]


def _sample(row: dict) -> tuple:
    return _unit(row)[:-1] + (row["sample_idx"],)


# ---------- stage 1: generate ----------

def generate(cfg: dict, tasks: dict, out: Path) -> Path:
    path = out / "generations.jsonl"
    done = {_unit(r) for r in read_jsonl(path)}
    proto = protocol(cfg)
    client = VllmClient(cfg["server"]["base_url"], cfg["server"]["model"], cfg["server"]["request_timeout_s"])

    units = []
    for (bench, tm, tid), task in tasks.items():
        for mode in cfg["modes"]:
            for decoding in cfg["decodings"]:
                plan = sampling_plan(proto, mode, decoding, cfg["seed"])
                for run_idx, sampling in enumerate(plan or []):
                    key = (bench, tm, mode, decoding, tid, run_idx)
                    if key not in done:
                        units.append((key, task, sampling, proto["modes"][mode]["enable_thinking"]))
    print(f"generate: {len(done)} requests done, {len(units)} to go")

    def one(key, task, sampling, thinking):
        bench, tm, mode, decoding, tid, run_idx = key
        messages = messages_for(task)
        rows = []
        for i, s in enumerate(client.generate(messages, sampling, thinking)):
            answer, reasoning = split_reasoning(s["content"], s["reasoning"])
            rows.append({"task_id": tid, "benchmark": bench, "task_mode": tm, "mode": mode, "decoding": decoding,
                         "run": run_idx, "sample_idx": run_idx * sampling["n"] + i,
                         "prompt": messages[-1]["content"], "generation": answer, "reasoning": reasoning,
                         "finish_reason": s["finish_reason"], "completion_tokens": s["completion_tokens"],
                         "served_model": cfg["server"]["model"], "seed": sampling["seed"]})
        return rows

    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(cfg["server"]["concurrency"]) as pool:
        futures = [pool.submit(one, *u) for u in units]
        for k, fut in enumerate(as_completed(futures), 1):
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fut.result()))   # one write per request
            f.flush()
            if k % 50 == 0 or k == len(futures):
                print(f"  {k}/{len(futures)} requests")
    return path


# ---------- stage 2: execute ----------

def execute_all(cfg: dict, tasks: dict, gen_path: Path, out: Path) -> Path:
    path = out / "results.jsonl"
    done = {_sample(r) for r in read_jsonl(path)}
    gens = [g for g in read_jsonl(gen_path) if _sample(g) not in done and (g["benchmark"], g["task_mode"], g["task_id"]) in tasks]
    print(f"execute: {len(done)} samples done, {len(gens)} to go")

    # Identical programs for the same task (e.g. greedy runs) are executed once.
    jobs: dict[tuple, list[dict]] = defaultdict(list)
    no_program = []
    for g in gens:
        task = tasks[(g["benchmark"], g["task_mode"], g["task_id"])]
        program = program_from_answer(task, g["generation"])
        g["program"] = program
        if program is None:
            no_program.append(g)
        else:
            jobs[(g["benchmark"], g["task_mode"], g["task_id"], hashlib.sha256(program.encode()).hexdigest())].append(g)

    def run_one(rows: list[dict]) -> list[dict]:
        g = rows[0]
        task = tasks[(g["benchmark"], g["task_mode"], g["task_id"])]
        r = execute(g["program"], task.tests, lang=task.lang, timeout_s=cfg["execute"]["timeout_s"])
        verdict = {"status": r.status.value, "compiled": r.compiled, "passed": r.passed,
                   "tests_passed": r.tests_passed, "tests_total": r.tests_total,
                   "stderr": r.compile_stderr or r.run_stderr, "duration_s": r.duration_s,
                   "cobc_version": r.meta.get("cobc_version", "")}
        return [{**x, **verdict} for x in rows]

    empty = {"status": "no_program", "compiled": False, "passed": False, "tests_passed": 0, "tests_total": 0,
             "stderr": "", "duration_s": 0.0, "cobc_version": ""}
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(cfg["execute"]["workers"]) as pool:
        f.write("".join(json.dumps({**g, **empty}, ensure_ascii=False) + "\n" for g in no_program))
        futures = [pool.submit(run_one, rows) for rows in jobs.values()]
        for k, fut in enumerate(as_completed(futures), 1):
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fut.result()))
            f.flush()
            if k % 100 == 0 or k == len(futures):
                print(f"  {k}/{len(futures)} programs")
    return path


# ---------- metrics ----------

def nondeterministic_tasks(bench: str) -> set[str]:
    known = benchmarks.manifest()[bench].get("canonical_known_failures", {})
    return {f"{bench}/{name}" for name, v in known.items() if v["kind"] == "nondeterministic"}


def compute_metrics(cfg: dict, results: list[dict]) -> dict:
    """{"<benchmark>/<task_mode>/<mode>/<decoding>": summary}; benchmarks with nondeterministic tasks
    also get a "<...>/deterministic" summary without them."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups["/".join((r["benchmark"], r["task_mode"], r["mode"], r["decoding"]))].append(r)
    out = {}
    for key in sorted(groups):
        out[key] = summarize(groups[key], cfg["ks"])
        skip = nondeterministic_tasks(key.split("/")[0])
        if skip:
            out[f"{key}/deterministic"] = summarize(groups[key], cfg["ks"], skip)
    return out


def print_metrics(metrics: dict) -> None:
    print(f"{'group':58} {'tasks':>5} {'pass@1':>7} {'pass@10':>7} {'compile':>7} {'trunc':>6}")
    for key, m in metrics.items():
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "-"
        print(f"{key:58} {m['n_tasks']:>5} {fmt(m.get('pass@1')):>7} {fmt(m.get('pass@10')):>7} "
              f"{fmt(m.get('compile_rate')):>7} {fmt(m.get('truncation_rate')):>6}")


# ---------- W&B ----------

TABLE_COLUMNS = ["task_id", "benchmark", "task_mode", "mode", "decoding", "sample_idx", "prompt", "generation",
                 "reasoning", "program", "status", "compiled", "tests_passed", "tests_total", "stderr",
                 "duration_s", "finish_reason"]


def _cell(row: dict, col: str):
    v = row.get(col)
    if isinstance(v, str):
        return v[:TABLE_REASONING if col == "reasoning" else TABLE_TEXT]
    return v


def log_results(run, results: list[dict], metrics: dict, out: Path) -> None:
    import wandb
    run.log({"generations": wandb.Table(columns=TABLE_COLUMNS,
                                        data=[[_cell(r, c) for c in TABLE_COLUMNS] for r in results])})
    flat = {f"{group}/{k}": v for group, m in metrics.items() for k, v in m.items() if not isinstance(v, dict)}
    run.summary.update(flat)
    versions = sorted({r["cobc_version"] for r in results if r.get("cobc_version")})
    run.summary["cobc_version"] = ", ".join(versions)
    art = wandb.Artifact("eval-results", type="eval-results", metadata={"metrics": metrics, "cobc_version": versions})
    art.add_file(str(out / "results.jsonl"))
    art.add_file(str(out / "metrics.json"))
    run.log_artifact(art, aliases=[run.name])


def log_generations(run, gen_path: Path, cfg: dict) -> None:
    import wandb
    art = wandb.Artifact("eval-generations", type="generations",
                         metadata={"model_name": cfg["model_name"], "served_model": cfg["server"]["model"],
                                   "adapter": cfg.get("adapter"), "run_name": cfg["run_name"]})
    art.add_file(str(gen_path))
    run.log_artifact(art, aliases=[cfg["run_name"]])


def run_config(cfg: dict) -> dict:
    pins = {name: {"revision": b["revision"], "files": b["fetch"]["files"]} for name, b in benchmarks.manifest().items()}
    return {**cfg, "git_commit": git_commit(), "protocol": protocol(cfg), "benchmark_pins": pins,
            "execute_backend": os.environ.get("EXECUTE_BACKEND", "docker"),
            "vast_instance": os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL")}


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Eval: generate -> execute -> metrics -> W&B.")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", choices=["all", "generate", "execute"], default="all")
    ap.add_argument("--generations", help="execute stage: W&B artifact (name:alias) or local generations.jsonl")
    ap.add_argument("--run-name", help="overrides run_name (and the output folder)")
    ap.add_argument("--limit", type=int, help="only the first N tasks of every benchmark / task mode")
    ap.add_argument("--modes", nargs="+", choices=["non_thinking", "thinking"],
                    help="overrides modes; same run_name later resumes with the other mode")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.run_name:
        cfg["run_name"] = a.run_name
    if a.modes:
        cfg["modes"] = a.modes
    out = Path(cfg["out_dir"]) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(cfg, a.limit)

    run = None
    if not a.no_wandb:
        import wandb
        job = {"all": "eval", "generate": "eval-generate", "execute": "eval"}[a.stage]
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "onprem-codegen"), job_type=job,
                         name=cfg["run_name"] if a.stage != "generate" else f"{cfg['run_name']}-generate",
                         config=run_config(cfg))
        for name in cfg["benchmarks"]:   # lineage: logs the version if it is new, else reuses it
            run.use_artifact(benchmarks.artifact(name))

    if a.stage in ("all", "generate"):
        gen_path = generate(cfg, tasks, out)
        if run:
            log_generations(run, gen_path, cfg)
    if a.stage in ("all", "execute"):
        if a.stage == "execute" and a.generations and not Path(a.generations).exists():
            if not run:
                raise SystemExit("--generations names a W&B artifact: drop --no-wandb or pass a local file")
            gen_path = Path(run.use_artifact(a.generations).download()) / "generations.jsonl"
        elif a.stage == "execute":
            gen_path = Path(a.generations) if a.generations else out / "generations.jsonl"
        results_path = execute_all(cfg, tasks, gen_path, out)
        results = read_jsonl(results_path)
        metrics = compute_metrics(cfg, results)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
        print_metrics(metrics)
        if run:
            log_results(run, results, metrics, out)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
