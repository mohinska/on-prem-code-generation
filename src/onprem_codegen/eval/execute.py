"""
onprem_codegen/eval/execute.py — execution contract for generated code.

The interface is final. EXECUTE_MODE=stub returns deterministic fake results;
EXECUTE_MODE=real compiles and runs the code through sandbox.py.

Usage:
    from onprem_codegen.eval.execute import execute, TestCase

    res = execute(code, tests=[TestCase(input="5\n", expected="25\n")], lang="cobol")
    if res.passed: ...

The mode is selected by the EXECUTE_MODE environment variable:
    stub  (default) — deterministic fake results, no cobc/docker
    real  — real execution in the eval-sandbox (see sandbox.py for EXECUTE_BACKEND)

What `tests` means (the type decides how the code is checked):
    list[TestCase]    stdin -> stdout, COBOL program is compiled with `cobc -x`
    list[CallerTest]  COBOLEval: a caller program CALLs the solution and writes the result to a file
    list[FileIOTest]  CobolCodeBench: input files in, output files compared exactly
    str               Python: a self-contained test program appended after `code` (it must
                      call its own check, e.g. a `def check(c): ...` followed by `check(my_func)`)
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal, TypedDict, Union

from onprem_codegen.eval import sandbox

Lang = Literal["cobol", "python"]


class Status(str, Enum):
    OK = "ok"                        # compiled and all tests passed
    COMPILE_ERROR = "compile_error"  # cobc / Python syntax check failed
    RUNTIME_ERROR = "runtime_error"  # program crashed at runtime
    TIMEOUT = "timeout"              # exceeded timeout_s
    WRONG_OUTPUT = "wrong_output"    # ran, but output didn't match expected


@dataclass(frozen=True)
class TestCase:
    """One COBOL test: what to feed on stdin and what to expect on stdout."""
    __test__ = False  # keep pytest from collecting this as a test class
    input: str
    expected: str


@dataclass(frozen=True)
class CallerTest:
    """COBOLEval test: `caller` is a COBOL main program that CALLs the solution and writes
    RESULT to `result_file`; the file is parsed by `type` and compared with `expected`."""
    caller: str
    expected: str       # Python literal of the expected value, e.g. "True", "[1, 2]"
    type: str           # Bool | Int | Float | String | List[Int] | List[Float] | List[String]
    result_file: str    # e.g. "HAS-CLOSE-ELEMENTS.TXT"


@dataclass(frozen=True)
class FileIOTest:
    """CobolCodeBench test: write `input_files` ({name: content}) into the working directory,
    run the program, then every file in `expected_outputs` must exist and match exactly."""
    input_files: dict
    expected_outputs: dict


# COBOL: a list of TestCase / CallerTest / FileIOTest (one kind per list).
# Python (HumanEval): a string of self-contained test code.
Tests = Union[list[TestCase], list[CallerTest], list[FileIOTest], str]


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
    meta: dict = field(default_factory=dict)  # e.g. {"stub": True, "cobc_version": ...}

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

    mode = os.environ.get("EXECUTE_MODE", "stub")
    if mode == "stub":
        return _execute_stub(code, tests, lang)
    if mode == "real":
        return _execute_real(code, tests, lang, timeout_s)
    raise ValueError(f"unknown EXECUTE_MODE: {mode}")


# ---------- real execution ----------

COMPILE_TIMEOUT_S = 30.0


def _build_job(code: str, tests: Tests, lang: Lang, timeout_s: float) -> dict:
    """Validates the (lang, tests) combination and builds the sandbox_runner job."""
    if lang == "python":
        if not isinstance(tests, str):
            raise ValueError("python tests must be a str (self-contained test program)")
        kind, payload = "python", tests
    else:
        kinds = {TestCase: "stdin", CallerTest: "caller", FileIOTest: "file_io"}
        if tests == []:   # contracts.md: no tests means compile only (cobc -fsyntax-only)
            kind, payload = "compile", []
        elif (isinstance(tests, str) or type(tests[0]) not in kinds
                or any(type(t) is not type(tests[0]) for t in tests)):
            raise ValueError("cobol tests must be a list of one test type: "
                             "TestCase, CallerTest or FileIOTest (or [] to only compile)")
        else:
            kind, payload = kinds[type(tests[0])], [asdict(t) for t in tests]
    return {"kind": kind, "code": code, "tests": payload,
            "timeout_s": timeout_s, "compile_timeout_s": COMPILE_TIMEOUT_S}


def _execute_real(code: str, tests: Tests, lang: Lang, timeout_s: float) -> ExecResult:
    job = _build_job(code, tests, lang, timeout_s)
    backend = os.environ.get("EXECUTE_BACKEND", "docker")
    total = len(tests) if isinstance(tests, list) else 1
    meta = {"stub": False, "lang": lang, "kind": job["kind"], "backend": backend}

    if not code.strip():   # nothing to compile; skip the container start
        return ExecResult(Status.COMPILE_ERROR, compiled=False, passed=False, tests_total=total,
                          compile_stderr="empty program", meta=meta)

    # Every COBOLEval test is compiled separately, hence one compile slot per test.
    n = max(total, 1)
    budget_s = COMPILE_TIMEOUT_S * n + timeout_s * n + 10.0
    raw = sandbox.run_job(job, budget_s, backend)
    meta["cobc_version"] = raw.get("cobc_version", "")
    return _result_from_raw(raw, total, meta)


def _t_empty() -> dict:
    return {"status": "ok", "stdout": "", "stderr": "", "duration_s": 0.0}


def _result_from_raw(raw: dict, total: int, meta: dict) -> ExecResult:
    if not raw["compiled"]:
        return ExecResult(Status.COMPILE_ERROR, compiled=False, passed=False, tests_total=total,
                          compile_stderr=raw.get("compile_stderr", ""), meta=meta)
    runs = raw["tests"]
    failed = [t for t in runs if t["status"] != "ok"]
    # show the first failure, else the last run (compile-only jobs have no runs)
    shown = failed[0] if failed else (runs[-1] if runs else _t_empty())
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


def _execute_stub(code: str, tests: Tests, lang: Lang) -> ExecResult:
    """Deterministic fake: same code -> same result.

    The status depends on the code hash, so C sees every branch
    (compile_error, wrong_output, ok, ...) and can test their handling.
    """
    total = len(tests) if isinstance(tests, list) else 1
    base = dict(tests_total=total, meta={"stub": True, "lang": lang})

    if not code.strip():
        return ExecResult(Status.COMPILE_ERROR, compiled=False, passed=False,
                          compile_stderr="stub: empty program", **base)

    bucket = int(hashlib.sha256(code.encode()).hexdigest(), 16) % 10
    if total == 0 and isinstance(tests, list):   # compile only: just compile_error or ok
        if bucket < 3:
            return ExecResult(Status.COMPILE_ERROR, compiled=False, passed=False,
                              compile_stderr="stub: syntax error", duration_s=0.05, **base)
        return ExecResult(Status.OK, compiled=True, passed=True, duration_s=0.05, **base)
    if bucket < 3:   # ~30%
        return ExecResult(Status.COMPILE_ERROR, compiled=False, passed=False,
                          compile_stderr="stub: syntax error", duration_s=0.05, **base)
    if bucket == 3:  # ~10%
        return ExecResult(Status.RUNTIME_ERROR, compiled=True, passed=False,
                          run_stderr="stub: runtime error", duration_s=0.1, **base)
    if bucket == 4:  # ~10%
        return ExecResult(Status.TIMEOUT, compiled=True, passed=False,
                          run_stderr="stub: timeout", duration_s=10.0, **base)
    if bucket < 7:   # ~20%
        return ExecResult(Status.WRONG_OUTPUT, compiled=True, passed=False,
                          tests_passed=total // 2, run_stdout="stub: wrong", duration_s=0.1,
                          **{k: v for k, v in base.items()})
    return ExecResult(Status.OK, compiled=True, passed=True, tests_passed=total,
                      run_stdout="stub: ok", duration_s=0.1, **base)  # ~30%


# ---------- docs/contracts.md section 2 API (used by data/rejection_sampling.py) ----------

class Result(TypedDict):
    status: Literal["ok", "compile_error", "runtime_error", "timeout", "test_fail"]
    stderr: str         # compiler stderr, or program stderr of the first failing test
    duration_s: float


_COBOL_BLOCK = re.compile(r"```cobol[^\n]*\n(.*?)```", re.S | re.I)


def extract_program(text: str) -> str | None:
    """First ```cobol fenced block in a model answer; None if there is none."""
    m = _COBOL_BLOCK.search(text)
    return m.group(1) if m else None


def run_program(source: str, tests: list[dict], timeout_s: float = 10.0) -> Result:
    """Compile `source` with GnuCOBOL and run it once per {"input", "output"} test (stdin -> stdout).
    tests=[] only checks that it compiles. `test_fail` is execute()'s wrong_output."""
    r = execute(source, [TestCase(t["input"], t["output"]) for t in tests], lang="cobol", timeout_s=timeout_s)
    status = "test_fail" if r.status == Status.WRONG_OUTPUT else r.status.value
    return {"status": status, "stderr": r.compile_stderr or r.run_stderr, "duration_s": r.duration_s}


if __name__ == "__main__":
    demo = execute("IDENTIFICATION DIVISION.", [TestCase("1\n", "1\n")])
    print(demo.to_dict())
