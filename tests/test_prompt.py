import pytest

from onprem_codegen.prompt import build_messages, build_prompt, instruction


def test_matches_contract_text():
    assert build_prompt("Add two numbers.") == (
        "Add two numbers.\n\n"
        "Write a complete GnuCOBOL program in fixed format. Read input from standard input and write the\n"
        "result to standard output. Answer with a single ```cobol code block."
    )


def test_strips_task_whitespace():
    assert build_prompt("\n  Task.\n\n") == f"Task.\n\n{instruction()}"


def test_empty_task_rejected():
    with pytest.raises(ValueError):
        build_prompt("   ")


def test_messages_single_user_turn_no_system():
    assert build_messages("T") == [{"role": "user", "content": build_prompt("T")}]
