"""
onprem_codegen/eval/execute.py: execution contract for generated code.

Compiles and runs code through sandbox.py (see EXECUTE_BACKEND there).

    res = execute(code, tests=[{"input": "5\\n", "expected": "25\\n"}], lang="cobol")
    if res.passed: ...

What `tests` means (the keys of the dicts decide how the code is checked):
    {"input", "expected"}                          stdin -> stdout, COBOL program is compiled with `cobc -x`
    {"caller", "expected", "type", "result_file"}  COBOLEval: a caller program CALLs the solution and writes the
                                                   result to `result_file`, parsed by `type` (Bool | Int | Float |
                                                   String | List[...]) and compared with the Python literal `expected`
    {"input_files", "expected_outputs"}            CobolCodeBench: {name: content} files in, files compared exactly
    []                                             COBOL: compile only (cobc -fsyntax-only)
    str                                            Python: a self-contained test program appended after `code` (it must
                                                   call its own check, e.g. a `def check(c): ...` then `check(my_func)`)
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal

from onprem_codegen.eval import sandbox

Lang = Literal["cobol", "python"]


class Status(str, Enum):
    OK = "ok"                        # compiled and all tests passed
    COMPILE_ERROR = "compile_error"  # cobc / Python syntax check failed
    RUNTIME_ERROR = "runtime_error"  # program crashed at runtime
    TIMEOUT = "timeout"              # exceeded timeout_s
    WRONG_OUTPUT = "wrong_output"    # ran, but output didn't match expected


Tests = list[dict] | str
KINDS = {"input": "stdin", "caller": "caller", "input_files": "file_io"}   # key of a test -> runner kind


@dataclass
class ExecResult:
    status: Status
    compiled: bool
    passed: bool
    tests_passed: int = 0
    tests_total: int = 0
    compile_stderr: str = ""
    run_stdout: str = ""
    run_stderr: str = ""
    duration_s: float = 0.0
    meta: dict = field(default_factory=dict)  # e.g. {"cobc_version": ...}

    def to_dict(self) -> dict:
        """Flat dict, ready to add as a row to a wandb.Table."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


def execute(code: str, tests: Tests, lang: Lang = "cobol", timeout_s: float = 10.0) -> ExecResult:
    """Compiles (for COBOL) and runs `code` on `tests`.

    Contract guarantees:
      * never raises because of bad code; errors go into status;
      * passed == True only if compiled == True and all tests passed;
      * compiled == False  =>  status == COMPILE_ERROR.
    """
    if lang not in ("cobol", "python"):
        raise ValueError(f"unsupported lang: {lang}")  # a caller error, not a code error

    job = _build_job(code, tests, lang, timeout_s)
    backend = os.environ.get("EXECUTE_BACKEND", "docker")
    total = len(tests) if isinstance(tests, list) else 1
    meta = {"lang": lang, "kind": job["kind"], "backend": backend}

    if not code.strip():   # nothing to compile; skip the container start
        return ExecResult(Status.COMPILE_ERROR, compiled=False, passed=False, tests_total=total,
                          compile_stderr="empty program", meta=meta)

    # Every COBOLEval test is compiled separately, hence one compile slot per test.
    n = max(total, 1)
    budget_s = COMPILE_TIMEOUT_S * n + timeout_s * n + 10.0
    raw = sandbox.run_job(job, budget_s, backend)
    meta["cobc_version"] = raw.get("cobc_version", "")
    return _result_from_raw(raw, total, meta)


COMPILE_TIMEOUT_S = 30.0


def _build_job(code: str, tests: Tests, lang: Lang, timeout_s: float) -> dict:
    """Validates the (lang, tests) combination and builds the sandbox_runner job."""
    if lang == "python":
        if not isinstance(tests, str):
            raise ValueError("python tests must be a str (self-contained test program)")
        kind, payload = "python", tests
    else:
        kinds = {next((v for k, v in KINDS.items() if k in t), None) for t in tests} if isinstance(tests, list) else {None}
        if len(kinds) > 1 or None in kinds:
            raise ValueError("cobol tests must be a list of one test type: stdin, caller or file_io (or [] to only compile)")
        kind = kinds.pop() if kinds else "compile"   # contracts.md: no tests means compile only (cobc -fsyntax-only)
        payload = tests
    return {"kind": kind, "code": code, "tests": payload,
            "timeout_s": timeout_s, "compile_timeout_s": COMPILE_TIMEOUT_S}


def _result_from_raw(raw: dict, total: int, meta: dict) -> ExecResult:
    if not raw["compiled"]:
        return ExecResult(Status.COMPILE_ERROR, compiled=False, passed=False, tests_total=total,
                          compile_stderr=raw.get("compile_stderr", ""), meta=meta)
    runs = raw["tests"]
    failed = [t for t in runs if t["status"] != "ok"]
    # show the first failure, else the last run (compile-only jobs have no runs)
    shown = failed[0] if failed else (runs[-1] if runs else {"stdout": "", "stderr": ""})
    return ExecResult(
        Status(shown["status"]) if failed else Status.OK,
        compiled=True,
        passed=not failed,
        tests_passed=len(runs) - len(failed),
        tests_total=total,
        compile_stderr=raw.get("compile_stderr", ""),
        run_stdout=shown["stdout"],
        run_stderr=shown["stderr"],
        duration_s=sum(t["duration_s"] for t in runs),
        meta=meta,
    )


_COBOL_BLOCK = re.compile(r"```cobol[^\n]*\n(.*?)```", re.S | re.I)


def extract_program(text: str) -> str | None:
    """First ```cobol fenced block in a model answer; None if there is none."""
    m = _COBOL_BLOCK.search(text)
    return m.group(1) if m else None
