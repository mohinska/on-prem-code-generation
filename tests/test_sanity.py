"""sanity.check(): known reference failures are expected, everything else must pass."""
from onprem_codegen.eval import sanity
from onprem_codegen.eval.benchmarks import Task
from onprem_codegen.eval.execute import ExecResult, Status

KNOWN = {"bad": {"kind": "broken_reference", "why": "x"}}


def _task(name):
    return Task(task_id=f"b/{name}", benchmark="b", mode="completion", lang="python", prompt="",
                entry_point=name, tests="", canonical_solution=f"REF {name}")


def _fake(results):
    """results: {(entry_point, is_reference): passed}"""
    def run(task, code, timeout_s):
        ok = results[(task.entry_point, code.startswith("REF"))]
        return ExecResult(Status.OK if ok else Status.WRONG_OUTPUT, compiled=True, passed=ok)
    return run


def test_known_failure_is_tolerated(monkeypatch):
    monkeypatch.setattr(sanity, "_run", _fake({("good", True): True, ("bad", True): False,
                                               ("good", False): False, ("bad", False): False}))
    assert sanity.check([_task("good"), _task("bad")], 1.0, KNOWN)


def test_unknown_failure_fails_the_check(monkeypatch):
    monkeypatch.setattr(sanity, "_run", _fake({("good", True): False, ("good", False): False}))
    assert not sanity.check([_task("good")], 1.0, KNOWN)


def test_known_failure_that_now_passes_fails_the_check(monkeypatch):
    monkeypatch.setattr(sanity, "_run", _fake({("bad", True): True, ("bad", False): False}))
    assert not sanity.check([_task("bad")], 1.0, KNOWN)


def test_negative_control_must_fail(monkeypatch):
    monkeypatch.setattr(sanity, "_run", _fake({("good", True): True, ("good", False): True}))
    assert not sanity.check([_task("good")], 1.0, KNOWN)
