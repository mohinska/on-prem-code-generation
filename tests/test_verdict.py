import json

from onprem_codegen.eval import verdict

N = 30  # tasks per benchmark: enough for a clear bootstrap CI


def rows(bench, decoding, passed, compiled=None, truncated=0.0, n=N, samples=None, task_mode="completion"):
    """`passed` / `compiled`: share of tasks whose every sample passes / compiles; `truncated`: share of tasks cut."""
    samples = samples or (20 if decoding == "sampled" else 3)
    compiled = passed if compiled is None else compiled
    out = []
    for t in range(n):
        for i in range(samples):
            out.append({"task_id": f"{bench}/{t}", "benchmark": bench, "task_mode": task_mode, "mode": "non_thinking",
                        "decoding": decoding, "sample_idx": i, "passed": t < passed * n, "compiled": t < compiled * n,
                        "status": "ok" if t < passed * n else "wrong_output",
                        "finish_reason": "length" if t < truncated * n else "stop"})
    return out


def run(cobol_sampled, compile_=None, cobol_greedy=None, python=0.95, ccb=0.0, ccb_trunc=0.0):
    r = rows("coboleval", "sampled", cobol_sampled, compile_)
    r += rows("coboleval", "greedy", cobol_greedy if cobol_greedy is not None else cobol_sampled, compile_)
    r += rows("humaneval_python", "sampled", python) + rows("humaneval_python", "greedy", python)
    r += rows("cobolcodebench", "sampled", ccb, truncated=ccb_trunc) + rows("cobolcodebench", "greedy", ccb, truncated=ccb_trunc)
    return r


BASE_V2 = [r for r in run(0.1, compile_=0.2) if r["benchmark"] != "humaneval_python"]   # v2 has no HumanEval
BASE_FROZEN = run(0.0, compile_=0.0)                                                   # old baseline: everything


def heldout(lo, mean=0.2):
    return {"delta": {"sampled": {"pass@1": {"mean": mean, "ci": [lo, mean + 0.1]}}}}


def judge(adapted, held=heldout(0.05), seed2=None, held2=None):
    return verdict.judge(verdict.merge_baseline(BASE_V2, BASE_FROZEN), adapted, held, seed2, held2)


def test_success_when_main_python_guard_and_training_works():
    v = judge(run(0.6, compile_=0.8))
    assert v["verdict"] == "успіх"
    assert all(c["met"] for c in v["criteria"])


def test_partial_when_training_works_but_coboleval_does_not_improve():
    v = judge(run(0.1, compile_=0.2))
    assert v["verdict"] == "частковий"
    assert not v["by_name"]["main"]["met"] and v["by_name"]["training_works"]["met"]


def test_failure_when_held_out_tasks_do_not_improve():
    v = judge(run(0.6, compile_=0.8), held=heldout(-0.02, mean=0.01))
    assert v["verdict"] == "невдача"


def test_main_needs_a_higher_compile_rate_too():
    v = judge(run(0.6, compile_=0.15))    # pass@1 up, but compile rate below the baseline 0.2
    assert not v["by_name"]["main"]["met"]


def test_python_guard_allows_two_points_and_no_more():
    base = verdict.merge_baseline(BASE_V2, run(0.0, compile_=0.0, python=0.9))    # HumanEval comes from the frozen run
    same = verdict.judge(base, run(0.6, compile_=0.8, python=0.9), heldout(0.05))
    assert same["by_name"]["python_guard"]["met"] and same["verdict"] == "успіх"
    worse = verdict.judge(base, run(0.6, compile_=0.8, python=0.8), heldout(0.05))   # -0.1
    assert not worse["by_name"]["python_guard"]["met"]
    assert worse["verdict"] == "успіх без guard Python"


def test_cobolcodebench_guard_fails_on_more_truncation():
    v = judge(run(0.6, compile_=0.8, ccb_trunc=0.2))
    assert not v["by_name"]["ccb_guard"]["met"]


def test_merge_takes_v2_groups_and_fills_the_rest_from_the_frozen_baseline():
    merged = verdict.merge_baseline(BASE_V2, BASE_FROZEN)
    groups = {verdict.group(r) for r in merged}
    assert "humaneval_python/completion/non_thinking/sampled" in groups
    cobol = [r for r in merged if verdict.group(r) == "coboleval/completion/non_thinking/sampled"]
    assert sum(r["passed"] for r in cobol) == 0.1 * N * 20                     # from v2, not from the frozen baseline


def test_unstable_when_the_two_seeds_disagree():
    v = judge(run(0.6, compile_=0.8), seed2=[r for r in run(0.1, compile_=0.2) if r["benchmark"] == "coboleval"],
              held2=heldout(0.05))
    assert v["unstable"] and "нестабільний" in verdict.render(v)


def test_cli_writes_markdown_and_json(tmp_path):
    def dump(name, data):
        p = tmp_path / name
        p.write_text("".join(json.dumps(r) + "\n" for r in data))
        return p
    (tmp_path / "heldout").mkdir()
    (tmp_path / "heldout" / "report_cobol.json").write_text(json.dumps(heldout(0.05)))
    verdict.main(["--base", str(dump("v2.jsonl", BASE_V2)), "--base-frozen", str(dump("b.jsonl", BASE_FROZEN)),
                  "--adapted", str(dump("a.jsonl", run(0.6, compile_=0.8))), "--heldout", str(tmp_path / "heldout"),
                  "--out", str(tmp_path / "verdict.md")])
    assert "успіх" in (tmp_path / "verdict.md").read_text()
    assert json.loads((tmp_path / "verdict.json").read_text())["verdict"] == "успіх"
