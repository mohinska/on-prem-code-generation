import hashlib

import pytest

from onprem_codegen.data.completion_rows import make_completion_rows
from onprem_codegen.prompt import IO_COMPLETE, build_prompt

PROG = ("       IDENTIFICATION DIVISION.\n       PROGRAM-ID. P.\n       DATA DIVISION.\n"
        "       WORKING-STORAGE SECTION.\n       01 X PIC 9.\n       PROCEDURE DIVISION.\n"
        "           DISPLAY X.\n           GOBACK.\n")
NO_DATA = "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. P.\n       PROCEDURE DIVISION.\n           GOBACK.\n"


def row(i, program=PROG, task="Print a digit."):
    c = f"```cobol\n{program}```"
    return {"prompt": build_prompt(f"{task} {i}"), "completion": c, "source": "rejection_sampling",
            "task_id": f"codenet/p{i:05d}", "content_sha256": hashlib.sha256(c.encode()).hexdigest(),
            "meta": {"generator_model": "m"}}


def is_completion(r):
    return IO_COMPLETE in r["prompt"]


def test_round_trip_indent_and_gobacks():
    rows = [row(i) for i in range(40)]
    out = make_completion_rows(rows, seed=7)
    comp = [r for r in out if is_completion(r)]
    assert comp
    for r in comp:
        prefix = r["prompt"].split("\n\n", 1)[1].rsplit("\n\n", 1)[0]
        assert prefix.startswith("       IDENTIFICATION DIVISION.\n")  # indent kept
        assert "DATA DIVISION" not in prefix
        body = r["completion"]
        answer = body.removeprefix("```cobol\n").removesuffix("```")
        assert answer.startswith("       DATA DIVISION.\n")
        assert prefix + "\n" + answer == PROG  # prefix + answer is the original byte for byte
        assert "GOBACK" in answer
        assert r["content_sha256"] == hashlib.sha256(body.encode()).hexdigest()
        assert r["prompt"].startswith("Print a digit. ")  # task text recovered from the stored prompt


def test_share_by_task_not_row_and_seeded():
    rows = [row(i) for i in range(100) for _ in range(3)]  # 3 rows per task
    out = make_completion_rows(rows, seed=7)
    by_task = {}
    for r in out:
        by_task.setdefault(r["task_id"], set()).add(is_completion(r))
    assert all(len(v) == 1 for v in by_task.values())
    assert 0.45 <= sum(next(iter(v)) for v in by_task.values()) / 100 <= 0.55
    assert out == make_completion_rows(list(reversed(rows)), seed=7)[::-1]  # independent of row order
    assert [is_completion(r) for r in out] != [is_completion(r) for r in make_completion_rows(rows, seed=8)]


def test_programs_without_data_division_stay_full_and_input_untouched():
    rows = [row(i, NO_DATA) for i in range(20)]
    before = [dict(r) for r in rows]
    assert make_completion_rows(rows, seed=7) == rows
    assert rows == before


def test_prompt_without_standard_instruction_is_rejected():
    rows = [{**row(i), "prompt": "something else"} for i in range(10)]
    with pytest.raises(ValueError):
        make_completion_rows(rows, seed=7)


def test_blank_lines_before_data_division_go_to_the_answer_so_the_round_trip_stays_exact():
    prog = PROG.replace("       DATA DIVISION.", "\n       DATA DIVISION.")
    out = make_completion_rows([row(i, prog) for i in range(10)], seed=7)
    for r in (r for r in out if is_completion(r)):
        prefix = r["prompt"].split("\n\n", 1)[1].rsplit("\n\n", 1)[0] + "\n"
        assert prefix + r["completion"].removeprefix("```cobol\n").removesuffix("```") == prog


def test_data_division_with_padded_spaces_or_lower_case_is_split_too():
    for header in ("       DATA                             DIVISION.\n", "       data division.\n"):
        prog = PROG.replace("       DATA DIVISION.\n", header)
        (out,) = make_completion_rows([row(0, prog)], seed=0, fraction=0.99)
        assert is_completion(out) and out["completion"].startswith(f"```cobol\n{header}")


def test_cli_writes_rows_and_stats_and_fails_on_benchmark_hit(tmp_path, monkeypatch):
    import json
    from onprem_codegen.data import completion_rows

    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(row(i)) + "\n" for i in range(20)))
    out = tmp_path / "out"
    monkeypatch.setattr(completion_rows, "decontaminate", lambda path, report, cfg: 0)

    assert completion_rows.main(["--input", str(src), "--out", str(out), "--seed", "7", "--no-wandb"]) == 0

    rows = [json.loads(l) for l in (out / "sft.jsonl").read_text().splitlines()]
    stats = json.loads((out / "stats.json").read_text())
    assert len(rows) == 20 and stats["rows"] == 20 and stats["seed"] == 7
    assert stats["completion_rows"] == sum(is_completion(r) for r in rows) == 10
    assert stats["completion_share"] == 0.5 and stats["decontam_hits"] == 0

    monkeypatch.setattr(completion_rows, "decontaminate", lambda path, report, cfg: 1)
    assert completion_rows.main(["--input", str(src), "--out", str(out), "--seed", "7", "--no-wandb"]) == 1
