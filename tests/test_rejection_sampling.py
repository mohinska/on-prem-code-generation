from pathlib import Path

import yaml

from onprem_codegen.data import rejection_sampling as rs
from onprem_codegen.eval.execute import ExecResult, Status
from onprem_codegen.prompt import build_prompt

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/data/rejection_sampling.yaml").read_text())
TASK = {"task_id": "codenet/p1", "prompt": "Print A+B.", "tests": [{"input": "3 5\n", "output": "8\n"}],
        "meta": {"problem_id": "p1"}}
PROG = "       PROCEDURE DIVISION.\n           DISPLAY \"OK\".\n"


def _sample(i, status="ok", program=PROG, hardcoded=False):
    return {"task_id": "codenet/p1", "sample_idx": i, "prompt": build_prompt("Print A+B."), "program": program,
            "status": status, "hardcoded": hardcoded}


def test_hardcoded_detection():
    assert rs.is_hardcoded('           DISPLAY "8".', TASK["tests"])
    assert rs.is_hardcoded("           DISPLAY 8.", TASK["tests"])
    assert not rs.is_hardcoded("           DISPLAY WS-SUM.", TASK["tests"])
    assert not rs.is_hardcoded('           DISPLAY "Yes".', [{"input": "", "output": "Yes\n"}])


def test_build_sft_keeps_only_passing_distinct_capped():
    samples = [_sample(0), _sample(1),  # same program twice
               _sample(2, "wrong_output", PROG + "*"), _sample(3, hardcoded=True, program=PROG + "**")]
    samples += [_sample(10 + i, program=PROG + f"      * v{i}\n") for i in range(6)]
    rows, dropped = rs.build_sft(samples, [], CFG)
    assert len(rows) == CFG["keep"]["max_per_task"]
    assert dropped == {"duplicate": 1, "hardcoded": 1, "over_cap": 3}
    r = rows[0]
    assert set(r) == {"prompt", "completion", "source", "task_id", "content_sha256", "meta"}
    assert r["completion"].startswith("```cobol\n       PROCEDURE") and r["completion"].endswith("\n```")
    assert r["source"] == "rejection_sampling"
    assert r["meta"] == {"generator_model": CFG["generator_model"], "submission_id": None}


class FakeClient:
    def generate(self, messages, sampling, thinking):
        return [{"content": f"```cobol\n{PROG}```", "finish_reason": "stop"}, {"content": "no code", "finish_reason": "stop"}]


def test_sample_task_statuses(monkeypatch):
    monkeypatch.setattr(rs, "run_program", lambda prog, tests, timeout: {"status": "ok", "stderr": "", "duration_s": 0.1})
    ok, none = rs.sample_task(TASK, FakeClient(), CFG)
    assert (ok["status"], none["status"], none["program"]) == ("ok", "no_program", None)
    assert ok["prompt"] == build_prompt("Print A+B.") and ok["hardcoded"] is False


def test_generation_hints_reach_the_model_but_not_the_saved_prompt(monkeypatch):
    monkeypatch.setattr(rs, "run_program", lambda prog, tests, timeout: {"status": "ok", "stderr": "", "duration_s": 0.1})
    seen = {}

    class Client(FakeClient):
        def generate(self, messages, sampling, thinking):
            seen["content"] = messages[0]["content"]
            return super().generate(messages, sampling, thinking)

    ok, _ = rs.sample_task(TASK, Client(), {**CFG, "generation_hints": "Use WS- names."})
    assert seen["content"] == build_prompt("Print A+B.") + "\n\nUse WS- names."
    assert ok["prompt"] == build_prompt("Print A+B.")


def test_worth_sampling_by_score():
    assert rs.worth_sampling({"prompt": "Score : 300 points\n\nProblem"}, 500)
    assert not rs.worth_sampling({"prompt": "Score : 600 points\n\nProblem"}, 500)
    assert not rs.worth_sampling({"prompt": "Reverse Sequence"}, 500)
    assert rs.worth_sampling({"prompt": "Reverse Sequence"}, None)


def test_select_todo_and_second_pass():
    def task(tid, score):
        return {"task_id": tid, "prompt": f"Score : {score} points\n\nProblem"}

    def samples(tid, n, status="wrong_output"):
        return [{"task_id": tid, "sample_idx": i, "status": status} for i in range(n)]

    tasks = {t["task_id"]: t for t in (task("new", 100), task("failed", 100), task("solved", 100), task("hard", 600),
                                       task("again", 100))}
    prior = samples("failed", 16) + samples("solved", 16, "ok") + samples("hard", 16) + samples("again", 32)
    cfg = {"sampling": {"n": 16}, "max_score": None, "second_pass_max_score": 300}
    assert [t["task_id"] for t in rs.select_todo(tasks, prior, cfg, False)] == ["new"]
    assert [t["task_id"] for t in rs.select_todo(tasks, prior, cfg, True)] == ["failed"]


def test_sample_task_start_idx(monkeypatch):
    monkeypatch.setattr(rs, "run_program", lambda prog, tests, timeout: {"status": "ok", "stderr": "", "duration_s": 0.1})
    assert [s["sample_idx"] for s in rs.sample_task(TASK, FakeClient(), CFG, 16)] == [16, 17]


def test_run_program_maps_tests_to_execute(monkeypatch):
    seen = {}

    def fake_execute(source, tests, timeout_s):
        seen.update(source=source, tests=tests)
        return ExecResult(Status.WRONG_OUTPUT, True, False, run_stderr="diff")

    monkeypatch.setattr(rs, "execute", fake_execute)
    r = rs.run_program("P", [{"input": "1\n", "output": "2\n"}], 5.0)
    assert r["status"] == "wrong_output" and r["stderr"] == "diff"
    assert seen["tests"] == [{"input": "1\n", "expected": "2\n"}]
