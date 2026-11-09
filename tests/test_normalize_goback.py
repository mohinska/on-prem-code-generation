import hashlib

from onprem_codegen.data.normalize_goback import build_v1_lite, stop_run_to_goback


def test_replaces_stop_run_statement_keeping_the_period():
    src = "       PROCEDURE DIVISION.\n           DISPLAY 'HI'\n           STOP RUN.\n"
    assert stop_run_to_goback(src) == (src.replace("STOP RUN", "GOBACK"), 1)


def test_is_case_insensitive_and_replaces_every_statement():
    src = "           IF N = 0\n               stop run\n           END-IF\n           Stop  Run.\n"
    out, n = stop_run_to_goback(src)
    assert n == 2 and "GOBACK" not in src and out.count("GOBACK") == 2 and "STOP" not in out.upper()


def test_leaves_comments_and_string_literals_alone():
    src = ("      * STOP RUN here is a comment\n"
           "           DISPLAY \"STOP RUN\"\n"
           "           DISPLAY 'STOP RUN' STOP RUN.\n")
    out, n = stop_run_to_goback(src)
    assert n == 1
    assert out == src[:-len("STOP RUN.\n")] + "GOBACK.\n"


def test_leaves_stop_run_with_a_status_clause():
    src = "           STOP RUN WITH ERROR STATUS 1.\n           STOP RUN RETURNING 2.\n"
    assert stop_run_to_goback(src) == (src, 0)


def test_program_without_stop_run_is_unchanged():
    src = "           GOBACK.\n"
    assert stop_run_to_goback(src) == (src, 0)


def row(task_id, program, source="codenet_cobol"):
    completion = f"```cobol\n{program}\n```"
    return {"prompt": f"prompt {task_id}", "completion": completion, "source": source, "task_id": task_id,
            "content_sha256": hashlib.sha256(completion.encode()).hexdigest(), "meta": {"submission_id": None}}


PROG = "       PROCEDURE DIVISION.\n           STOP RUN."


def test_build_v1_lite_drops_heldout_rewrites_and_rechecks():
    rows = [row("t/1", PROG), row("t/2", PROG), row("t/3", PROG), row("t/4", "           GOBACK.")]
    seen = []

    def passes(r, program):
        seen.append((r["task_id"], program))
        return r["task_id"] != "t/3"          # the rewritten program of t/3 fails its tests

    out, stats = build_v1_lite(rows, heldout={"t/2"}, passes=passes)

    assert [r["task_id"] for r in out] == ["t/1", "t/4"]
    assert out[0]["completion"] == "```cobol\n       PROCEDURE DIVISION.\n           GOBACK.\n```"
    assert out[0]["content_sha256"] == hashlib.sha256(out[0]["completion"].encode()).hexdigest()
    assert out[0]["prompt"] == "prompt t/1" and out[0]["source"] == "codenet_cobol"
    assert all("STOP RUN" not in p for _, p in seen)                    # the check runs on the new program
    assert [t for t, _ in seen] == ["t/1", "t/3", "t/4"]                # held-out rows are never executed
    assert stats == {"input": 4, "heldout": 1, "to_fixed": 0, "stop_run_replaced": 2, "no_goback": 0,
                     "failed_recheck": 1, "kept": 2}


def test_program_that_never_ends_explicitly_is_dropped_without_running():
    ran = []
    out, stats = build_v1_lite([row("t/1", "           DISPLAY 'X'.")], heldout=set(), passes=lambda r, p: ran.append(p) or True)
    assert out == [] and ran == [] and stats["no_goback"] == 1 and stats["kept"] == 0


def test_free_format_program_is_moved_to_fixed_format_and_rechecked():
    seen = []
    out, stats = build_v1_lite([row("t/1", "PROCEDURE DIVISION.\n    STOP RUN.")], heldout=set(),
                               passes=lambda r, p: seen.append(p) or True)
    assert seen == ["       PROCEDURE DIVISION.\n           GOBACK.\n"]      # the tests run on the fixed version
    assert out[0]["completion"] == "```cobol\n       PROCEDURE DIVISION.\n           GOBACK.\n```"
    assert stats["to_fixed"] == 1
