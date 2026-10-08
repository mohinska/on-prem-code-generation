"""Перевірки контракту onprem_codegen/eval/execute.py. Мають проходити і для stub, і для real-реалізації."""
from onprem_codegen.eval.execute import ExecResult, Status, TestCase, execute

TESTS = [TestCase(input="1\n", expected="1\n")]


def test_returns_exec_result():
    assert isinstance(execute("IDENTIFICATION DIVISION.", TESTS), ExecResult)


def test_invariants_hold():
    for i in range(200):
        r = execute(f"program {i}", TESTS)
        assert not (r.passed and not r.compiled)          # passed => compiled
        assert (not r.compiled) == (r.status == Status.COMPILE_ERROR)
        assert r.passed == (r.status == Status.OK)


def test_bad_code_never_raises():
    for code in ["", "   ", "garbage !!!", "\x00"]:
        execute(code, TESTS)  # не має кидати виняток


def test_empty_program_is_compile_error():
    assert execute("", TESTS).status == Status.COMPILE_ERROR


def test_python_mode_accepts_check_string():
    r = execute("def f(): return 1", "def check(c): assert c() == 1", lang="python")
    assert r.tests_total >= 1


def test_to_dict_is_flat_for_wandb():
    d = execute("x", TESTS).to_dict()
    assert isinstance(d["status"], str)
