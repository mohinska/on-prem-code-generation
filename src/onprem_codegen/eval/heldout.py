"""
onprem_codegen/eval/heldout.py: unbiased test on the held-out 20% of task-pool (stdin -> stdout CodeNet tasks).

    # on the instance next to the server (both models are served: base, cobol); resumable
    python -m onprem_codegen.eval.heldout generate --models base cobol
    # locally, in the sandbox
    python -m onprem_codegen.eval.heldout execute --models base cobol

Same prompt as SFT data and eval (`build_prompt`, no generation_hints), non-thinking, greedy x1 and sampled n 5
with the settings of evaluation_protocol. Output per model: <out>/<model>/generations.jsonl (rows like
eval-generations) and results.jsonl (+ verdict). Report per adapter: report_<model>.json, pass@1 / pass@5 and the
paired bootstrap CI of <model> - base.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from onprem_codegen.data.rejection_sampling import all_tests
from onprem_codegen.eval.execute import execute, extract_program
from onprem_codegen.eval.generate import VllmClient
from onprem_codegen.eval.metrics import delta_ci, task_scores
from onprem_codegen.eval.run import protocol
from onprem_codegen.prompt import build_messages
from onprem_codegen.utils import read_jsonl

POOL = Path("artifacts/task-pool:v1/task_pool.jsonl")
IDS = Path("data/splits/heldout_task_ids.json")
OUT = Path("outputs/eval/heldout")
BASE_URL = "http://localhost:8000/v1"
SEED, SAMPLED_N, TIMEOUT_S = 1234, 5, 10.0
KS = {"greedy": [1], "sampled": [1, SAMPLED_N]}


def load_tasks(pool: Path = POOL, ids: Path = IDS) -> dict[str, dict]:
    keep = set(json.loads(Path(ids).read_text(encoding="utf-8"))["task_ids"])
    return {t["task_id"]: t for t in read_jsonl(pool) if t["task_id"] in keep}


# ---------- stage 1: generate ----------

def generate(tasks: dict, model: str, base_url: str, out: Path, concurrency: int = 8) -> Path:
    path = Path(out) / "generations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["decoding"]) for r in read_jsonl(path)}   # a request writes all its samples at once
    m = protocol()["modes"]["non_thinking"]
    plan = {"greedy": {**m["greedy"], "max_tokens": m["max_new_tokens"], "seed": SEED},
            "sampled": {**m["sampled"], "n": SAMPLED_N, "max_tokens": m["max_new_tokens"], "seed": SEED}}
    client = VllmClient(base_url, model, 3600)
    todo = [(tid, dec) for tid in tasks for dec in plan if (tid, dec) not in done]
    print(f"generate {model}: {len(done)} requests done, {len(todo)} to go")

    def one(tid, dec):
        messages = build_messages(tasks[tid]["prompt"])
        return [{"task_id": tid, "benchmark": "codenet_heldout", "task_mode": "stdin", "mode": "non_thinking",
                 "decoding": dec, "run": 0, "sample_idx": i, "prompt": messages[0]["content"],
                 "generation": s["content"], "finish_reason": s["finish_reason"],
                 "completion_tokens": s["completion_tokens"], "served_model": model, "seed": SEED}
                for i, s in enumerate(client.generate(messages, plan[dec], False))]

    failed = 0
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(concurrency) as pool:
        futures = {pool.submit(one, *u): u for u in todo}
        for fut in as_completed(futures):
            try:
                rows = fut.result()
            except Exception as e:   # skip it, a re-run retries the missing request
                failed += 1
                print(f"  failed {futures[fut]}: {type(e).__name__}: {e}")
                continue
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            f.flush()
    if failed:
        print(f"generate {model}: {failed} requests failed; run the same command again to retry them")
    return path


# ---------- stage 2: execute ----------

def sandbox_executor(program: str, tests: list[dict]) -> dict:
    r = execute(program, [{"input": t["input"], "expected": t["output"]} for t in tests], timeout_s=TIMEOUT_S)
    return {"status": r.status.value, "compiled": r.compiled, "passed": r.passed}


def execute_all(tasks: dict, gen_path: Path, out: Path, executor=sandbox_executor, workers: int = 4) -> Path:
    path = Path(out) / "results.jsonl"
    key = lambda r: (r["task_id"], r["decoding"], r["sample_idx"])
    done = {key(r) for r in read_jsonl(path)}
    jobs: dict[tuple, list[dict]] = {}   # identical programs of a task run once
    rows = []
    for g in read_jsonl(gen_path):
        if key(g) in done or g["task_id"] not in tasks:
            continue
        program = extract_program(g["generation"])
        if program is None:
            rows.append({**g, "program": None, "status": "no_program", "compiled": False, "passed": False})
        else:
            jobs.setdefault((g["task_id"], program), []).append({**g, "program": program})
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        futures = {pool.submit(executor, program, all_tests(tasks[tid])): group for (tid, program), group in jobs.items()}
        for fut in as_completed(futures):
            f.write("".join(json.dumps({**g, **fut.result()}, ensure_ascii=False) + "\n" for g in futures[fut]))
            f.flush()
    return path


# ---------- report ----------

def report(base: list[dict], adapted: list[dict], adapted_name: str = "cobol") -> dict:
    """pass@k per model and decoding; delta = adapted - base with the paired bootstrap 95% CI over tasks."""
    rows = {"base": base, adapted_name: adapted}
    scores = {m: {d: {k: task_scores([r for r in rs if r["decoding"] == d], k) for k in KS[d]} for d in KS}
              for m, rs in rows.items()}
    mean = lambda s: sum(s.values()) / len(s)
    out = {m: {d: {f"pass@{k}": mean(scores[m][d][k]) for k in KS[d]} for d in KS} for m in rows}
    out["delta"] = {}
    for d in KS:
        out["delta"][d] = {}
        for k in KS[d]:
            b, a = scores["base"][d][k], scores[adapted_name][d][k]
            common = b.keys() & a.keys()
            out["delta"][d][f"pass@{k}"] = {"mean": mean({t: a[t] - b[t] for t in common}),
                                            "ci": list(delta_ci(b, a))}
    out["n_tasks"] = len(scores["base"]["greedy"][1].keys() & scores[adapted_name]["greedy"][1].keys())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out CodeNet tasks: generate / execute / report.")
    ap.add_argument("stage", choices=["generate", "execute"])
    ap.add_argument("--models", nargs="+", default=["base", "cobol"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--ids", type=Path, default=IDS)
    a = ap.parse_args()
    tasks = load_tasks(a.pool, a.ids)
    print(f"{len(tasks)} held-out tasks")
    for m in a.models:
        if a.stage == "generate":
            generate(tasks, m, a.base_url, a.out / m)
        else:
            execute_all(tasks, a.out / m / "generations.jsonl", a.out / m)
    if a.stage == "execute" and "base" in a.models:
        for m in (m for m in a.models if m != "base"):   # one report per adapter (cobol, cobol-seed2) against base
            rep = report(read_jsonl(a.out / "base" / "results.jsonl"), read_jsonl(a.out / m / "results.jsonl"), m)
            (a.out / f"report_{m}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
            print(m, json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
