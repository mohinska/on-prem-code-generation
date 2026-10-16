"""Contract checks for onprem_codegen/eval/execute.py (Python snippets on the local backend: no cobc or docker needed)."""
import pytest

from onprem_codegen.eval.execute import ExecResult, Status, execute

CHECK = "def check(c):\n    assert c() == 1\n\ncheck(f)\n"


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    monkeypatch.setenv("EXECUTE_BACKEND", "local")


def test_returns_exec_result():
    assert isinstance(execute("def f(): return 1", CHECK, lang="python"), ExecResult)


def test_invariants_hold():
    for code in ["def f(): return 1", "def f(): return 2", "def f(: pass", "def f(): raise SystemExit(3)"]:
        r = execute(code, CHECK, lang="python")
        assert not (r.passed and not r.compiled)          # passed => compiled
        assert (not r.compiled) == (r.status == Status.COMPILE_ERROR)
        assert r.passed == (r.status == Status.OK)


def test_bad_code_never_raises():
    for code in ["", "   ", "garbage !!!", "\x00"]:
        execute(code, CHECK, lang="python")  # must not raise


def test_to_dict_is_flat_for_wandb():
    assert isinstance(execute("def f(): return 1", CHECK, lang="python").to_dict()["status"], str)
