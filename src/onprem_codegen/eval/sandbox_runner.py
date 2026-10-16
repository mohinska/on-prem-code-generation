"""
onprem_codegen/eval/sandbox_runner.py: runs ONE execution job and writes result.json.

Stdlib only, on purpose: it is mounted read-only into the eval-sandbox container
(docker/eval-sandbox.Dockerfile) and must not depend on the rest of the package.

    python sandbox_runner.py /work/job.json     # writes /work/result.json next to the job

The job format is built by onprem_codegen.eval.execute; the kinds are:
    stdin    cobc -x, feed each test's stdin, compare stdout
    compile  tests == []: cobc -fsyntax-only, no run
    caller   COBOLEval: cobc -x caller solution, read <ENTRY>.TXT, parse by type
    file_io  CobolCodeBench: write input files, run, compare output files exactly
    python   HumanEval: the candidate code plus a self-contained test program
"""

from __future__ import annotations

import ast
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX_CAPTURE = 10_000   # chars of stdout/stderr kept per process

# One flag set per protocol, taken from the reference harnesses (see data/manifests/benchmarks.yaml).
CALLER_FLAGS = ["-w", "-fformat=variable", "-x"]   # COBOLEval scripts/evaluation.py
PLAIN_FLAGS = ["-x"]                               # CobolCodeBench compile_execute.py; stdin tasks


# ---------- process helper ----------

def _clip(s: str | bytes | None) -> str:
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="replace")
    return s[:MAX_CAPTURE]


def run(cmd: list[str], cwd: Path, timeout: float, stdin: str = "") -> dict:
    """Runs cmd with a hard timeout; the whole process group is killed on timeout."""
    t0 = time.monotonic()
    # Minimal environment: generated code gets no tokens or other secrets.
    env = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8", "HOME": str(cwd)}
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, text=True, errors="replace",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as e:   # e.g. cobc is not installed
        return {"returncode": 127, "timed_out": False, "stdout": "", "stderr": f"{e}", "duration_s": 0.0}
    timed_out = False
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, err = proc.communicate()
    return {"returncode": proc.returncode, "timed_out": timed_out,
            "stdout": _clip(out), "stderr": _clip(err), "duration_s": time.monotonic() - t0}


def _test_result(status: str, proc: dict | None = None, **extra) -> dict:
    d = {"status": status, "stdout": "", "stderr": "", "duration_s": 0.0}
    if proc:
        d.update(stdout=proc["stdout"], stderr=proc["stderr"], duration_s=proc["duration_s"])
    d.update(extra)
    return d


def _run_status(proc: dict) -> str | None:
    """Failure status of a finished process, None if it exited cleanly."""
    if proc["timed_out"]:
        return "timeout"
    if proc["returncode"] != 0:
        return "runtime_error"
    return None


def _cobc_version() -> str:
    p = run(["cobc", "--version"], Path.cwd(), 10)
    return p["stdout"].splitlines()[0] if p["stdout"] else ""


# ---------- COBOLEval result parsing (mirrors scripts/evaluation.py of the reference harness) ----------

def _parse_scalar(kind: str, line: str):
    line = line.strip()
    if kind == "Bool":
        return line == "1"
    if kind == "String":
        return line
    neg = line[:1] in ("p", "y")   # the reference harness writes negative numbers with a p/y prefix
    body = line[1:] if neg else line
    value = int(body) if kind == "Int" else float(body)
    return -value if neg else value


def parse_result(lines: list[str], type_: str, true):
    """Parses the lines of the caller's result file into a Python value of the declared type."""
    if type_.startswith("List[") and type_.endswith("]"):
        inner = type_[5:-1]
        return [_parse_scalar(inner, x) for x in lines][: len(true)]
    if type_ in ("Bool", "Int", "Float", "String"):
        return _parse_scalar(type_, lines[0])
    raise ValueError(f"invalid result type: {type_}")


def is_equal(type_: str, result, true) -> bool:
    if type_ == "Float":
        return math.isclose(result, true, abs_tol=0.001)
    if type_ == "List[Float]":
        return len(result) == len(true) and all(math.isclose(r, t, abs_tol=0.001) for r, t in zip(result, true))
    return result == true


def _norm_stdout(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").rstrip().split("\n"))


# ---------- kinds ----------

def _compile_cobol(sources: list[str], exe: str, flags: list[str], workdir: Path, timeout: float) -> dict:
    return run(["cobc", *flags, "-o", exe, *sources], workdir, timeout)


def _compile_failed(proc: dict) -> bool:
    return proc["timed_out"] or proc["returncode"] != 0


def _compile_and_run(job: dict, workdir: Path, run_test) -> dict:
    """One `cobc -x` build, then `run_test(i, t) -> (process, status)` per test."""
    (workdir / "solution.cbl").write_text(job["code"], encoding="utf-8")
    c = _compile_cobol(["solution.cbl"], "prog", PLAIN_FLAGS, workdir, job["compile_timeout_s"])
    if _compile_failed(c):
        return {"compiled": False, "compile_stderr": c["stderr"], "tests": []}
    return {"compiled": True, "compile_stderr": c["stderr"],
            "tests": [_test_result(status, p) for p, status in (run_test(i, t) for i, t in enumerate(job["tests"]))]}


def kind_stdin(job: dict, workdir: Path) -> dict:
    def run_test(i, t):
        p = run([str(workdir / "prog")], workdir, job["timeout_s"], stdin=t["input"])
        status = _run_status(p)
        if status is None:
            status = "ok" if _norm_stdout(p["stdout"]) == _norm_stdout(t["expected"]) else "wrong_output"
        return p, status
    return _compile_and_run(job, workdir, run_test)


def kind_caller(job: dict, workdir: Path) -> dict:
    (workdir / "solution.cbl").write_text(job["code"], encoding="utf-8")
    results = []
    for t in job["tests"]:
        (workdir / "caller.cbl").write_text(t["caller"], encoding="utf-8")
        c = _compile_cobol(["caller.cbl", "solution.cbl"], "call_prog", CALLER_FLAGS, workdir, job["compile_timeout_s"])
        if _compile_failed(c):
            return {"compiled": False, "compile_stderr": c["stderr"], "tests": []}
        result_file = workdir / t["result_file"]
        result_file.unlink(missing_ok=True)
        p = run([str(workdir / "call_prog")], workdir, job["timeout_s"])
        status = _run_status(p)
        if status is None:
            status = _check_caller_output(t, result_file)
        results.append(_test_result(status, p))
        result_file.unlink(missing_ok=True)
    return {"compiled": True, "compile_stderr": "", "tests": results}


def _check_caller_output(t: dict, result_file: Path) -> str:
    try:
        lines = result_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return "wrong_output"
        true = ast.literal_eval(t["expected"])
        if isinstance(true, tuple):
            true = list(true)
        parsed = parse_result(lines, t["type"], true)
        return "ok" if is_equal(t["type"], parsed, true) else "wrong_output"
    except FileNotFoundError:
        return "wrong_output"   # the program ran but never wrote its result
    except (ValueError, SyntaxError, IndexError):
        return "wrong_output"   # unparsable result file


def _safe_name(name: str) -> str:
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"unsafe file name in task: {name!r}")
    return name


def kind_file_io(job: dict, workdir: Path) -> dict:
    def run_test(i, t):
        rundir = workdir / f"run{i}"
        rundir.mkdir()
        for name, content in t["input_files"].items():
            (rundir / _safe_name(name)).write_text(content, encoding="utf-8")
        p = run([str(workdir / "prog")], rundir, job["timeout_s"])
        status = _run_status(p)
        if status is None:
            ok = True
            for name, expected in t["expected_outputs"].items():
                path = rundir / _safe_name(name)
                ok = ok and path.is_file() and path.read_text(encoding="utf-8", errors="replace") == expected
            status = "ok" if ok else "wrong_output"
        return p, status
    return _compile_and_run(job, workdir, run_test)


def kind_python(job: dict, workdir: Path) -> dict:
    try:
        compile(job["code"], "<solution>", "exec")   # syntax check only, nothing is executed
    except (SyntaxError, ValueError) as e:
        return {"compiled": False, "compile_stderr": f"{type(e).__name__}: {e}", "tests": []}
    (workdir / "prog.py").write_text(job["code"] + "\n\n" + job["tests"], encoding="utf-8")
    p = run([sys.executable, "-I", "prog.py"], workdir, job["timeout_s"])
    status = _run_status(p)
    if status == "runtime_error" and "AssertionError" in p["stderr"]:
        status = "wrong_output"
    return {"compiled": True, "compile_stderr": "", "tests": [_test_result(status or "ok", p)]}


def kind_compile(job: dict, workdir: Path) -> dict:
    """tests == []: syntax check only, nothing is linked or run."""
    (workdir / "solution.cbl").write_text(job["code"], encoding="utf-8")
    c = run(["cobc", "-fsyntax-only", "solution.cbl"], workdir, job["compile_timeout_s"])
    return {"compiled": not _compile_failed(c), "compile_stderr": c["stderr"], "tests": []}


KINDS = {"compile": kind_compile, "stdin": kind_stdin, "caller": kind_caller, "file_io": kind_file_io, "python": kind_python}


def main(job_path: str) -> None:
    job_file = Path(job_path)
    job = json.loads(job_file.read_text(encoding="utf-8"))
    workdir = job_file.parent
    out = KINDS[job["kind"]](job, workdir)
    if job["kind"] != "python":
        out["cobc_version"] = _cobc_version()
    (workdir / "result.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
