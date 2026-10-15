"""
onprem_codegen/eval/sanity.py — model-free check that the execution harness works.

    python -m onprem_codegen.eval.sanity --bench humaneval_python
    python -m onprem_codegen.eval.sanity --bench cobolcodebench --mode completion
    python -m onprem_codegen.eval.sanity --bench humaneval_python --backend local --limit 20

Two controls per benchmark, no LLM involved:
    positive  the canonical solution of every task must pass (expected 100%)
    negative  a program that solves nothing must fail every task (expected 0%)
If either is off, the harness (compiler flags, test parsing, sandbox) is broken and any model
number produced with it is meaningless. Exit code 1 on failure.

COBOLEval ships no COBOL reference solutions, so only the negative control is possible for it;
validate its caller protocol on a machine with cobc via a known-good program instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

from onprem_codegen.eval.benchmarks import Task, load
from onprem_codegen.eval.execute import execute

NOTHING = {
    "cobol": "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. NOTHING.\n"
             "       PROCEDURE DIVISION.\n           GOBACK.\n       END PROGRAM NOTHING.\n",
    "python": "pass\n",
}


def _run(task: Task, code: str, timeout_s: float):
    return execute(code, task.tests, lang=task.lang, timeout_s=timeout_s)


def check(tasks: list[Task], timeout_s: float) -> bool:
    ok = True
    with_ref = [t for t in tasks if t.canonical_solution]
    if with_ref:
        results = [_run(t, t.canonical_solution, timeout_s) for t in with_ref]
        passed = sum(r.passed for r in results)
        print(f"positive control: {passed}/{len(with_ref)} canonical solutions pass")
        print(f"  statuses: {dict(Counter(r.status.value for r in results))}")
        for t, r in zip(with_ref, results):
            if not r.passed:
                print(f"  FAIL {t.task_id} [{r.status.value}] {(r.compile_stderr or r.run_stderr)[:200]!r}")
        ok &= passed == len(with_ref)
    else:
        print("positive control: skipped (no reference solution in this benchmark's language)")

    results = [_run(t, NOTHING[t.lang], timeout_s) for t in tasks]
    passed = sum(r.passed for r in results)
    print(f"negative control: {passed}/{len(tasks)} empty programs pass (expected 0)")
    ok &= passed == 0
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Harness sanity check on canonical solutions.")
    ap.add_argument("--bench", required=True, choices=["coboleval", "cobolcodebench", "humaneval_python"])
    ap.add_argument("--mode", default=None, help="completion | instruct (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--backend", choices=["docker", "local"], default=None,
                    help="default: $EXECUTE_BACKEND or docker. `local` is not a sandbox")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    os.environ["EXECUTE_MODE"] = "real"
    if args.backend:
        os.environ["EXECUTE_BACKEND"] = args.backend

    tasks = load(args.bench, args.mode)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{args.bench}: {len(tasks)} tasks, backend={os.environ.get('EXECUTE_BACKEND', 'docker')}")
    sys.exit(0 if check(tasks, args.timeout) else 1)


if __name__ == "__main__":
    main()
