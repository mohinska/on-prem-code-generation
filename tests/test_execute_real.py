"""EXECUTE_MODE=real: sandbox runner, result mapping and test-type validation.

Python tests use the `local` backend with trusted inline snippets, so they need neither docker
nor cobc. COBOL execution tests use a native cobc if there is one, else the eval-sandbox docker
image, and are skipped when neither is available.
"""
import shutil
import subprocess

import pytest

from onprem_codegen.eval import execute as execute_mod
from onprem_codegen.eval import sandbox, sandbox_runner
from onprem_codegen.eval.execute import CallerTest, FileIOTest, Status, TestCase, execute

CHECK = "def check(c):\n    assert c(2) == 4\n\ncheck(double)\n"


def _cobol_backend() -> str | None:
    if shutil.which("cobc"):
        return "local"
    try:
        found = subprocess.run(["docker", "image", "inspect", sandbox.DEFAULT_IMAGE],
                               capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        found = False
    return "docker" if found else None


COBOL_BACKEND = _cobol_backend()


@pytest.fixture(autouse=True)
def real_local(monkeypatch):
    monkeypatch.setenv("EXECUTE_MODE", "real")
    monkeypatch.setenv("EXECUTE_BACKEND", "local")


# ---------- python kind ----------

def test_python_pass():
    r = execute("def double(x):\n    return x * 2\n", CHECK, lang="python")
    assert (r.status, r.compiled, r.passed, r.tests_passed, r.tests_total) == (Status.OK, True, True, 1, 1)
    assert r.meta["stub"] is False


def test_python_wrong_output():
    r = execute("def double(x):\n    return x + 3\n", CHECK, lang="python")
    assert r.status == Status.WRONG_OUTPUT and r.compiled and not r.passed
    assert "AssertionError" in r.run_stderr


def test_python_syntax_error_is_compile_error():
    r = execute("def double(x) return x", CHECK, lang="python")
    assert r.status == Status.COMPILE_ERROR and not r.compiled and not r.passed
    assert "SyntaxError" in r.compile_stderr


def test_python_runtime_error():
    r = execute("def double(x):\n    raise RuntimeError('boom')\n", CHECK, lang="python")
    assert r.status == Status.RUNTIME_ERROR and r.compiled and not r.passed


def test_python_timeout():
    r = execute("def double(x):\n    while True:\n        pass\n", CHECK, lang="python", timeout_s=1.0)
    assert r.status == Status.TIMEOUT and r.compiled and not r.passed
    assert r.duration_s < 5


@pytest.mark.parametrize("code", ["", "   \n", "\x00", "garbage !!!"])
def test_bad_code_never_raises_and_is_compile_error(code):
    r = execute(code, CHECK, lang="python")
    assert r.status in (Status.COMPILE_ERROR, Status.RUNTIME_ERROR)
    assert not r.passed


def test_empty_program_skips_the_sandbox(monkeypatch):
    monkeypatch.setattr(sandbox, "run_job", lambda *a, **k: pytest.fail("sandbox must not start"))
    assert execute("", [TestCase("", "")], lang="cobol").status == Status.COMPILE_ERROR


# ---------- (lang, tests) validation: caller errors raise ----------

def test_cobol_with_check_string_is_rejected():
    with pytest.raises(ValueError):
        execute("x", CHECK, lang="cobol")


def test_python_with_list_is_rejected():
    with pytest.raises(ValueError):
        execute("x", [TestCase("", "")], lang="python")


def test_mixed_test_types_are_rejected():
    with pytest.raises(ValueError):
        execute("x", [TestCase("", ""), FileIOTest({}, {})], lang="cobol")


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("EXECUTE_BACKEND", "nope")
    with pytest.raises(ValueError):
        execute("def double(x): return x", CHECK, lang="python")


# ---------- result mapping ----------

def _t(status, out="", err="", dur=0.5):
    return {"status": status, "stdout": out, "stderr": err, "duration_s": dur}


def test_result_all_ok():
    raw = {"compiled": True, "tests": [_t("ok"), _t("ok")]}
    r = execute_mod._result_from_raw(raw, 2, {})
    assert (r.status, r.passed, r.tests_passed, r.tests_total, r.duration_s) == (Status.OK, True, 2, 2, 1.0)


def test_result_partial_shows_first_failure():
    raw = {"compiled": True, "tests": [_t("ok"), _t("wrong_output", out="bad"), _t("timeout")]}
    r = execute_mod._result_from_raw(raw, 3, {})
    assert (r.status, r.passed, r.tests_passed, r.run_stdout) == (Status.WRONG_OUTPUT, False, 1, "bad")


def test_result_compile_error():
    r = execute_mod._result_from_raw({"compiled": False, "compile_stderr": "boom", "tests": []}, 3, {})
    assert (r.status, r.compiled, r.passed, r.compile_stderr) == (Status.COMPILE_ERROR, False, False, "boom")


# ---------- COBOLEval result parsing ----------

@pytest.mark.parametrize("lines,type_,true,expected", [
    (["1\n"], "Bool", True, True),
    (["0"], "Bool", True, False),
    (["0042"], "Int", 42, 42),
    (["p0007"], "Int", -7, -7),             # negative numbers use a p/y prefix
    ([" hello "], "String", "hello", "hello"),
    (["1", "2", "3", "4"], "List[Int]", [1, 2, 3], [1, 2, 3]),   # extra lines are cut to len(true)
    (["y1.5"], "Float", -1.5, -1.5),
])
def test_parse_result(lines, type_, true, expected):
    assert sandbox_runner.parse_result(lines, type_, true) == expected


def test_parse_result_rejects_unknown_type():
    with pytest.raises(ValueError):
        sandbox_runner.parse_result(["1"], "Dict", 1)


def test_is_equal_float_tolerance():
    assert sandbox_runner.is_equal("Float", 1.0004, 1.0)
    assert not sandbox_runner.is_equal("Float", 1.01, 1.0)
    assert sandbox_runner.is_equal("List[Float]", [1.0, 2.0004], [1.0, 2.0])
    assert not sandbox_runner.is_equal("List[Float]", [1.0], [1.0, 2.0])


def test_stdout_comparison_ignores_trailing_whitespace_only():
    n = sandbox_runner._norm_stdout
    assert n("a  \r\nb\n\n") == n("a\nb")
    assert n("a b") != n("a  b")


def test_unsafe_input_file_names_are_rejected():
    for bad in ["../x", "a/b", "", ".."]:
        with pytest.raises(ValueError):
            sandbox_runner._safe_name(bad)


# ---------- docker backend wiring (no daemon needed) ----------

def test_docker_args_are_locked_down(tmp_path):
    args = sandbox._docker_args(tmp_path, "n", "img")
    joined = " ".join(args)
    for flag in ["--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "--pids-limit", "--memory"]:
        assert flag in joined
    assert ":/opt/sandbox_runner.py:ro" in joined


def test_missing_docker_is_a_sandbox_error_not_a_verdict(monkeypatch):
    monkeypatch.setenv("EXECUTE_BACKEND", "docker")

    def no_docker(*a, **k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", no_docker)
    with pytest.raises(sandbox.SandboxError):
        execute("def double(x): return x", CHECK, lang="python")


# ---------- COBOL end to end (needs cobc) ----------

needs_cobc = pytest.mark.skipif(COBOL_BACKEND is None,
                                reason="needs a native cobc or the eval-sandbox docker image")


@pytest.fixture
def cobol_backend(monkeypatch):
    monkeypatch.setenv("EXECUTE_BACKEND", COBOL_BACKEND)

HELLO = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO.
       PROCEDURE DIVISION.
           DISPLAY "HELLO".
           STOP RUN.
"""

ADD_ONE = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. ADD-ONE.
       DATA DIVISION.
       LINKAGE SECTION.
       01 LINKED-ITEMS.
           05 L-N PIC 9(4).
           05 RESULT PIC 9(4).
       PROCEDURE DIVISION USING LINKED-ITEMS.
           COMPUTE RESULT = L-N + 1.
           GOBACK.
       END PROGRAM ADD-ONE.
"""

ADD_ONE_CALLER = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. ADD-ONE-CALL.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT OUTPUT-FILE ASSIGN TO "ADD-ONE.TXT"
               ORGANIZATION IS LINE SEQUENTIAL.
       DATA DIVISION.
       FILE SECTION.
       FD OUTPUT-FILE.
       01 OUTPUT-RECORD PIC 9(4).
       WORKING-STORAGE SECTION.
       01 LINKED-ITEMS.
           05 L-N PIC 9(4).
           05 RESULT PIC 9(4).
       PROCEDURE DIVISION.
           MOVE 41 TO L-N
           CALL "ADD-ONE" USING LINKED-ITEMS
           OPEN OUTPUT OUTPUT-FILE
           MOVE RESULT TO OUTPUT-RECORD
           WRITE OUTPUT-RECORD
           CLOSE OUTPUT-FILE
           STOP RUN.
"""


@needs_cobc
def test_cobol_stdout_pass_and_fail(cobol_backend):
    assert execute(HELLO, [TestCase("", "HELLO\n")], lang="cobol").passed
    r = execute(HELLO, [TestCase("", "BYE\n")], lang="cobol")
    assert r.status == Status.WRONG_OUTPUT and r.compiled


@needs_cobc
def test_cobol_syntax_error_is_compile_error(cobol_backend):
    r = execute("THIS IS NOT COBOL", [TestCase("", "")], lang="cobol")
    assert r.status == Status.COMPILE_ERROR and not r.compiled
    assert r.meta["cobc_version"]


@needs_cobc
def test_cobol_caller_protocol(cobol_backend):
    ok = CallerTest(ADD_ONE_CALLER, "42", "Int", "ADD-ONE.TXT")
    bad = CallerTest(ADD_ONE_CALLER, "43", "Int", "ADD-ONE.TXT")
    assert execute(ADD_ONE, [ok], lang="cobol").passed
    r = execute(ADD_ONE, [ok, bad], lang="cobol")
    assert (r.status, r.tests_passed, r.tests_total) == (Status.WRONG_OUTPUT, 1, 2)


# ---------- docs/contracts.md section 2 API ----------

def test_extract_program_takes_first_cobol_block():
    text = "hi\n```cobol\nA\n```\n```cobol\nB\n```"
    assert execute_mod.extract_program(text) == "A\n"
    assert execute_mod.extract_program("```cobol\nunclosed") is None
    assert execute_mod.extract_program("```python\nx\n```") is None


def _fake_run_job(raw, seen):
    def run_job(job, budget_s, backend=None):
        seen.append(job)
        return raw
    return run_job


def test_run_program_maps_wrong_output_to_test_fail(monkeypatch):
    seen = []
    raw = {"compiled": True, "tests": [_t("wrong_output", err="diff")]}
    monkeypatch.setattr(sandbox, "run_job", _fake_run_job(raw, seen))
    r = execute_mod.run_program("PROGRAM", [{"input": "1\n", "output": "2\n"}])
    assert r["status"] == "test_fail" and r["stderr"] == "diff"
    assert seen[0]["kind"] == "stdin" and seen[0]["tests"] == [{"input": "1\n", "expected": "2\n"}]


def test_run_program_without_tests_only_compiles(monkeypatch):
    seen = []
    monkeypatch.setattr(sandbox, "run_job", _fake_run_job({"compiled": True, "tests": []}, seen))
    r = execute_mod.run_program("PROGRAM", [])
    assert r["status"] == "ok" and seen[0]["kind"] == "compile"


def test_compile_only_failure_is_compile_error(monkeypatch):
    raw = {"compiled": False, "compile_stderr": "syntax", "tests": []}
    monkeypatch.setattr(sandbox, "run_job", _fake_run_job(raw, []))
    r = execute("PROGRAM", [], lang="cobol")
    assert (r.status, r.compiled, r.passed, r.tests_total) == (Status.COMPILE_ERROR, False, False, 0)


def test_stub_compile_only_never_runs_the_program(monkeypatch):
    monkeypatch.setenv("EXECUTE_MODE", "stub")
    statuses = {execute(f"program {i}", [], lang="cobol").status for i in range(60)}
    assert statuses <= {Status.OK, Status.COMPILE_ERROR} and len(statuses) == 2


@needs_cobc
def test_cobol_compile_only_real(cobol_backend):
    assert execute(HELLO, [], lang="cobol").passed
    assert execute("THIS IS NOT COBOL", [], lang="cobol").status == Status.COMPILE_ERROR
