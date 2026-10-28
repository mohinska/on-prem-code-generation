"""Benchmark manifest and normalisers. Offline: the loader tests only run when the pinned files
are already in benchmarks/ (python -m onprem_codegen.eval.benchmarks downloads them)."""
import json
import re

import pytest

from onprem_codegen.eval import benchmarks as b


def test_manifest_pins_every_fetchable_benchmark():
    entries = b.manifest()
    assert set(entries) == {"coboleval", "cobolcodebench", "humaneval_python"}
    for name, e in entries.items():
        assert re.fullmatch(r"[0-9a-f]{40}", e["revision"]), name
        assert e["fetch"]["source"] in b.URL_TEMPLATES, name
        assert e["n_tasks"] > 0 and e["license"], name
        assert set(e["fetch"]["files"]) <= {"completion", "instruct"}, name
        for f in e["fetch"]["files"].values():
            assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]), name


def test_normalize_coboleval():
    rows = [{"task_id": "HumanEval/7", "prompt": "P", "entry_point": "has_close_elements",
             "canonical_solution": "    return True\n",
             "tests": [{"test": "CALLER", "result": {"value": "[1, 2]", "type_": {"List": "Int"}}},
                       {"test": "CALLER2", "result": {"value": "True", "type_": "Bool"}}]}]
    [t] = b.normalize_coboleval(rows, "completion")
    assert t.task_id == "coboleval/HumanEval/7" and t.lang == "cobol" and t.canonical_solution is None
    assert t.tests == [
        {"caller": "CALLER", "expected": "[1, 2]", "type": "List[Int]", "result_file": "HAS-CLOSE-ELEMENTS.TXT"},
        {"caller": "CALLER2", "expected": "True", "type": "Bool", "result_file": "HAS-CLOSE-ELEMENTS.TXT"}]


def test_normalize_coboleval_rejects_non_literal_expected_value():
    row = {"task_id": "x", "prompt": "", "entry_point": "f",
           "tests": [{"test": "", "result": {"value": "__import__('os')", "type_": "Int"}}]}
    with pytest.raises(ValueError):
        b.normalize_coboleval([row], "completion")


def test_normalize_cobolcodebench_modes():
    row = {"program_name": "task_func_01", "complete_prompt": "SKELETON", "instruct_prompt": "WRITE A PROGRAM",
           "inputs": json.dumps({"in.ps": "a\n"}), "outputs": json.dumps({"out.ps": "b\n"}),
           "canonical_solution": "FULL PROGRAM"}
    [c] = b.normalize_cobolcodebench([row], "completion")
    [i] = b.normalize_cobolcodebench([row], "instruct")
    assert c.prompt == "SKELETON" and i.prompt == "WRITE A PROGRAM"
    assert c.tests == [{"input_files": {"in.ps": "a\n"}, "expected_outputs": {"out.ps": "b\n"}}]
    assert c.canonical_solution == "FULL PROGRAM" and c.entry_point == "task_func_01"


def test_normalize_humaneval_is_self_contained():
    row = {"task_id": "HumanEval/0", "prompt": "def f(x):\n    '''doc'''\n", "entry_point": "f",
           "canonical_solution": "    return x\n", "test": "def check(candidate):\n    assert candidate(1) == 1\n"}
    [t] = b.normalize_humaneval([row], "completion")
    assert t.tests.rstrip().endswith("check(f)")
    assert t.canonical_solution == row["prompt"] + row["canonical_solution"]
    ns: dict = {}
    exec(t.canonical_solution + "\n" + t.tests, ns)   # the assembled program really runs


def _have(name: str) -> bool:
    spec = b.manifest()[name]["fetch"]
    return all(b.local_path(name, f).is_file() for f in spec["files"].values())


@pytest.mark.parametrize("name,n,modes", [
    ("coboleval", 146, {"completion"}),
    ("cobolcodebench", 46, {"completion", "instruct"}),
    ("humaneval_python", 164, {"completion"}),
])
def test_load_pinned_benchmark(name, n, modes):
    if not _have(name):
        pytest.skip("pinned files not downloaded")
    tasks = b.load(name)
    assert {t.mode for t in tasks} == modes
    assert len(tasks) == n * len(modes)
    assert all(t.prompt and t.entry_point and t.tests for t in tasks)


def test_known_failures_are_well_formed():
    for name, e in b.manifest().items():
        for task, v in e.get("canonical_known_failures", {}).items():
            assert v["kind"] in {"nondeterministic", "broken_reference"} and v["why"], (name, task)


def test_artifact_names_and_files():
    pytest.importorskip("wandb")
    for name, e in b.manifest().items():
        try:
            art = b.artifact(name)
        except (FileNotFoundError, OSError):
            pytest.skip(f"{name} not fetched")
        assert art.name == e["wandb_artifact"] and art.type == "benchmark"
        assert art.metadata["revision"] == e["revision"] and art.metadata["role"] == "eval_only"
        assert len(art.manifest.entries) == len(e["fetch"]["files"])
