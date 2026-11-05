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


def _sample(i, status="ok", program=PROG, hardcoded=False, n_tests=2, task_text="Print A+B."):
    return {"task_id": "codenet/p1", "sample_idx": i, "prompt": build_prompt(task_text), "program": program,
            "status": status, "hardcoded": hardcoded, "n_tests": n_tests}


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


def test_build_sft_drops_few_tests_and_too_hard():
    samples = [_sample(0, n_tests=1), _sample(1, task_text="Score : 600 points\n\nHard"),
               _sample(2, program=PROG + "*", task_text="Score : 500 points\n\nFine"), _sample(3, program=PROG + "**", task_text="No score")]
    rows, dropped = rs.build_sft(samples, [], CFG)
    assert len(rows) == 2 and dropped == {"few_tests": 1, "too_hard": 1}


def test_all_tests_adds_the_other_statement_samples():
    prompt = ("Print twice.\n\nSample Input 1\n\n1\n2\n\nSample Output 1\n\n2\n\nOne way is this.\n\n"
              "Sample Input 2\n\n3\n\nSample Output 2\n\n6\n7")
    task = {"prompt": prompt, "tests": [{"input": "1\n2\n", "output": "2\n"}]}
    assert rs.all_tests(task) == [{"input": "1\n2\n", "output": "2\n"}, {"input": "3\n", "output": "6\n7\n"}]


def test_all_tests_aoj_headers():
    prompt = "Sample Input 1\n\n1\n\nOutput for the Sample Input 1\n\n2\n\nSample Input 2\n\n3\n\nOutput for the Sample Input 2\n\n4"
    assert len(rs.all_tests({"prompt": prompt, "tests": [{"input": "1\n", "output": "2\n"}]})) == 2


def test_all_tests_keeps_the_task_test_if_parsing_disagrees():
    prompt = "Sample Input 1\n\n1\n\nSample Output 1\n\n9\n\nSample Input 2\n\n3\n\nSample Output 2\n\n6"
    tests = [{"input": "1\n", "output": "2\n"}]
    assert rs.all_tests({"prompt": prompt, "tests": tests}) == tests
    assert rs.all_tests(TASK) == TASK["tests"]


def test_recheck_ok_runs_each_sample_once(monkeypatch):
    prompt = "Sample Input 1\n\n3 5\n\nSample Output 1\n\n8\n\nSample Input 2\n\n1 1\n\nSample Output 2\n\n2"
    task = {**TASK, "prompt": prompt}
    ran = []

    def fake_run(prog, tests, timeout):
        ran.append(prog)
        return {"status": "wrong_output", "stderr": "", "duration_s": 0.1}

    monkeypatch.setattr(rs, "run_program", fake_run)
    samples = [{"task_id": "codenet/p1", "program": "OLD", "status": "ok"},
               {"task_id": "codenet/p1", "program": "NEW", "status": "ok", "n_tests": 2},
               {"task_id": "codenet/p1", "program": "BAD", "status": "compile_error"},
               {"task_id": "other", "program": "OTHER", "status": "ok"}]
    assert rs.recheck_ok(samples, {"codenet/p1": task}, {**CFG}) == 1
    assert ran == ["OLD"]
    assert [s["status"] for s in samples] == ["wrong_output", "ok", "compile_error", "ok"] and samples[0]["n_tests"] == 2
    assert rs.recheck_ok(samples, {"codenet/p1": task}, {**CFG}) == 0


def test_passes_ignoring_spacing(monkeypatch):
    results = {"sp": ExecResult(Status.WRONG_OUTPUT, True, False, run_stdout="   8\n"),
               "bad": ExecResult(Status.WRONG_OUTPUT, True, False, run_stdout="9\n"),
               "ce": ExecResult(Status.COMPILE_ERROR, False, False),
               "ok": ExecResult(Status.OK, True, True)}
    monkeypatch.setattr(rs, "execute", lambda source, tests, timeout_s: results[source])
    assert rs.passes_ignoring_spacing("sp", TASK["tests"], 5.0)
    assert rs.passes_ignoring_spacing("ok", TASK["tests"], 5.0)
    assert not rs.passes_ignoring_spacing("bad", TASK["tests"], 5.0)
    assert not rs.passes_ignoring_spacing("ce", TASK["tests"], 5.0)


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
