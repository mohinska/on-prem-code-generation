import pytest

from onprem_codegen.prompt import build_messages, build_prompt, instruction


def test_matches_contract_text():
    assert build_prompt("Add two numbers.") == (
        "Add two numbers.\n\n"
        "Write a complete GnuCOBOL program in fixed format. Read input from standard input with ACCEPT and write the "
        "result to standard output with DISPLAY. Answer with a single ```cobol code block."
    )


def test_drops_blank_lines_around_task_but_keeps_first_line_indent():
    assert build_prompt("\n  Task.\n\n") == f"  Task.\n\n{instruction()}"
    assert build_prompt(" \t\n\n  Task.  \n") == f"  Task.\n\n{instruction()}"


def test_fixed_format_cobol_prompt_keeps_its_7_space_indent():
    skeleton = "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. F.\n"
    assert build_prompt(skeleton).startswith("       IDENTIFICATION DIVISION.\n")


def test_empty_task_rejected():
    with pytest.raises(ValueError):
        build_prompt("   ")


def test_messages_single_user_turn_no_system():
    assert build_messages("T") == [{"role": "user", "content": build_prompt("T")}]
