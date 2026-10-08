"""
onprem_codegen/eval/execute.py — контракт виконання згенерованого коду.

СТАТУС: ЗАГЛУШКА (stub). Інтерфейс фінальний, реалізація — ні.
Справжня компіляція/запуск у sandbox з'явиться пізніше; сигнатура
execute() і структура ExecResult НЕ змінюватимуться.

Використання:
    from onprem_codegen.eval.execute import execute, TestCase

    res = execute(code, tests=[TestCase(input="5\n", expected="25\n")], lang="cobol")
    if res.passed: ...

Режим вибирається змінною середовища EXECUTE_MODE:
    stub  (за замовчуванням) — детерміновані фейкові результати, без cobc/docker
    real  — справжнє виконання в sandbox (поки NotImplementedError)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal, Union

Lang = Literal["cobol", "python"]


class Status(str, Enum):
    OK = "ok"                        # скомпілювалось і всі тести пройшли
    COMPILE_ERROR = "compile_error"  # cobc / синтаксис Python впав
    RUNTIME_ERROR = "runtime_error"  # програма впала під час виконання
    TIMEOUT = "timeout"              # перевищено timeout_s
    WRONG_OUTPUT = "wrong_output"    # виконалось, але вивід не збігся з expected


@dataclass(frozen=True)
class TestCase:
    """Один тест для COBOL: що подати на stdin і що очікувати на stdout."""
    __test__ = False  # не збирати pytest-ом як тестовий клас
    input: str
    expected: str


# Для COBOL — список TestCase.
# Для Python (HumanEval) — рядок з кодом тестів (функція check(...) з датасету).
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
    meta: dict = field(default_factory=dict)  # напр. {"stub": True, "cobc_version": ...}

    def to_dict(self) -> dict:
        """Плоский dict — зручно класти рядком у wandb.Table."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


def execute(code: str, tests: Tests, lang: Lang = "cobol", timeout_s: float = 10.0) -> ExecResult:
    """Компілює (для COBOL) і запускає `code` на `tests`.

    Гарантії контракту:
      * ніколи не кидає винятків через поганий код — помилки йдуть у status;
      * passed == True тільки якщо compiled == True і всі тести пройшли;
      * compiled == False  =>  status == COMPILE_ERROR.
    """
    if lang not in ("cobol", "python"):
        raise ValueError(f"unsupported lang: {lang}")  # це помилка виклику, не коду

    mode = os.environ.get("EXECUTE_MODE", "stub")
    if mode == "stub":
        return _execute_stub(code, tests, lang)
    if mode == "real":
        raise NotImplementedError("Sandbox execution is not implemented yet")
    raise ValueError(f"unknown EXECUTE_MODE: {mode}")


def _execute_stub(code: str, tests: Tests, lang: Lang) -> ExecResult:
    """Детермінований фейк: однаковий код -> однаковий результат.

    Розподіл статусів залежить від хешу коду, тож C бачить усі гілки
    (compile_error, wrong_output, ok, ...) і може тестувати свою обробку.
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
