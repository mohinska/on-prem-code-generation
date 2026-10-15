"""
onprem_codegen/eval/benchmarks.py — fetch pinned benchmarks and load them as `Task` objects.

    python -m onprem_codegen.eval.benchmarks                  # fetch + verify all, print a summary
    python -m onprem_codegen.eval.benchmarks --only coboleval

Revisions and sha256 come from data/manifests/benchmarks.yaml (the single source of truth).
Files land in benchmarks/<name>/ (gitignored); a hash mismatch is a hard error. These are
eval_only sources: nothing here may feed a training set.

    from onprem_codegen.eval.benchmarks import load
    for task in load("humaneval_python"):
        res = execute(code, task.tests, lang=task.lang)
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from onprem_codegen.eval.execute import CallerTest, FileIOTest, Lang, Tests

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "data" / "manifests" / "benchmarks.yaml"
BENCH_DIR = ROOT / "benchmarks"

URL_TEMPLATES = {
    "github": "https://raw.githubusercontent.com/{repo}/{revision}/{path}",
    "huggingface": "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}",
}


@dataclass
class Task:
    task_id: str                  # "<benchmark>/<original id>"; unique within (benchmark, mode)
    benchmark: str
    mode: str                     # "completion" | "instruct"
    lang: Lang
    prompt: str                   # what the model sees
    entry_point: str              # function name / PROGRAM-ID
    tests: Tests                  # ready to pass to execute()
    canonical_solution: str | None  # full reference program in `lang`, None if the benchmark has none
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- manifest + fetching ----------

def manifest() -> dict[str, dict]:
    doc = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return {b["name"]: b for b in doc["benchmarks"] if b.get("fetch")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str) -> dict[str, Path]:
    """Downloads the pinned files of `name` (skipped when already present) and verifies sha256.
    Returns {mode: local path}."""
    entry = manifest()[name]
    spec = entry["fetch"]
    out: dict[str, Path] = {}
    for mode, f in spec["files"].items():
        dest = BENCH_DIR / name / Path(f["path"]).name
        if not dest.is_file() or _sha256(dest) != f["sha256"]:
            url = URL_TEMPLATES[spec["source"]].format(repo=spec["repo"], revision=entry["revision"], path=f["path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "onprem-codegen"})
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
            actual = _sha256(dest)
            if actual != f["sha256"]:
                dest.unlink()
                raise ValueError(f"{name}/{mode}: sha256 mismatch for {url}\n"
                                 f"  expected {f['sha256']}\n  got      {actual}")
        out[mode] = dest
    return out


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------- normalisation: one function per benchmark ----------

def _result_type(t) -> str:
    """COBOLEval type_ is "Int" or {"List": "Int"}; we use "Int" / "List[Int]"."""
    if isinstance(t, dict) and len(t) == 1 and "List" in t:
        return f"List[{t['List']}]"
    if isinstance(t, str):
        return t
    raise ValueError(f"unsupported COBOLEval result type: {t!r}")


def normalize_coboleval(rows: list[dict], mode: str) -> list[Task]:
    out = []
    for r in rows:
        result_file = f"{r['entry_point'].upper().replace('_', '-')}.TXT"
        tests = []
        for t in r["tests"]:
            ast.literal_eval(t["result"]["value"])   # fail at load time, not mid-evaluation
            tests.append(CallerTest(caller=t["test"], expected=t["result"]["value"],
                                    type=_result_type(t["result"]["type_"]), result_file=result_file))
        out.append(Task(
            task_id=f"coboleval/{r['task_id']}", benchmark="coboleval", mode=mode, lang="cobol",
            prompt=r["prompt"], entry_point=r["entry_point"], tests=tests,
            canonical_solution=None,   # the dataset's reference solution is Python, not COBOL
        ))
    return out


def normalize_cobolcodebench(rows: list[dict], mode: str) -> list[Task]:
    prompt_key = {"completion": "complete_prompt", "instruct": "instruct_prompt"}[mode]
    out = []
    for r in rows:
        out.append(Task(
            task_id=f"cobolcodebench/{r['program_name']}", benchmark="cobolcodebench", mode=mode, lang="cobol",
            prompt=r[prompt_key], entry_point=r["program_name"],
            tests=[FileIOTest(input_files=json.loads(r["inputs"] or "{}"),
                              expected_outputs=json.loads(r["outputs"] or "{}"))],
            canonical_solution=r["canonical_solution"],
        ))
    return out


def normalize_humaneval(rows: list[dict], mode: str) -> list[Task]:
    return [Task(
        task_id=f"humaneval_python/{r['task_id']}", benchmark="humaneval_python", mode=mode, lang="python",
        prompt=r["prompt"], entry_point=r["entry_point"],
        # `test` defines check(candidate); append the call so the program is self-contained.
        tests=r["test"] + f"\n\ncheck({r['entry_point']})\n",
        canonical_solution=r["prompt"] + r["canonical_solution"],   # the full function
    ) for r in rows]


NORMALIZERS = {
    "coboleval": normalize_coboleval,
    "cobolcodebench": normalize_cobolcodebench,
    "humaneval_python": normalize_humaneval,
}


def load(name: str, mode: str | None = None) -> list[Task]:
    """All tasks of benchmark `name` (every mode unless `mode` is given). Fetches on first use."""
    entry = manifest()[name]
    n_expected = entry["n_tasks"]
    tasks: list[Task] = []
    for m, path in fetch(name).items():
        if mode and m != mode:
            continue
        rows = NORMALIZERS[name](_read_jsonl(path), m)
        if len(rows) != n_expected or len({t.task_id for t in rows}) != len(rows):
            raise ValueError(f"{name}/{m}: expected {n_expected} unique tasks, got {len(rows)}")
        tasks.extend(rows)
    if not tasks:
        raise ValueError(f"{name}: no tasks for mode={mode!r}")
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--only", nargs="*", help="benchmark names (default: all fetchable)")
    args = ap.parse_args()
    for name in args.only or manifest():
        tasks = load(name)
        modes = sorted({t.mode for t in tasks})
        print(f"ok {name}: {len(tasks)} tasks, modes={modes}, lang={tasks[0].lang}")


if __name__ == "__main__":
    main()
