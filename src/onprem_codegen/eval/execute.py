"""
onprem_codegen/eval/execute.py — execution contract for generated code.

STATUS: STUB. The interface is final; the implementation is not.
Real compilation/execution in the sandbox will come later; the
execute() signature and the ExecResult structure will NOT change.

Usage:
    from onprem_codegen.eval.execute import execute, TestCase

    res = execute(code, tests=[TestCase(input="5\n", expected="25\n")], lang="cobol")
    if res.passed: ...

The mode is selected by the EXECUTE_MODE environment variable:
    stub  (default) — deterministic fake results, no cobc/docker
    real  — real execution in the sandbox (NotImplementedError for now)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal, Union

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


# For COBOL: a list of TestCase.
# For Python (HumanEval): a string of test code (the check(...) function from the dataset).
Tests = Union[list[TestCase], str]


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
        raise NotImplementedError("Sandbox execution is not implemented yet")
    raise ValueError(f"unknown EXECUTE_MODE: {mode}")


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


if __name__ == "__main__":
    demo = execute("IDENTIFICATION DIVISION.", [TestCase("1\n", "1\n")])
    print(demo.to_dict())
