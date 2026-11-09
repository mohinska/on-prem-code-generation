from onprem_codegen.eval.repair_format import repair

PROMPT = ("       IDENTIFICATION DIVISION.\n       PROGRAM-ID. T.\n       DATA DIVISION.\n       WORKING-STORAGE SECTION.\n"
          "\n\nWrite a complete GnuCOBOL program in fixed format. Complete the program above. Answer with a single "
          "```cobol code block.")
BODY = "       PROCEDURE DIVISION.\n           GOBACK.\n"
FULL = "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. T.\n       DATA DIVISION.\n       WORKING-STORAGE SECTION.\n" + BODY


def test_good_program_is_unchanged():
    assert repair(FULL, PROMPT) == (FULL, None)


def test_first_line_from_column_one_gets_its_indent():
    assert repair(FULL.removeprefix("       "), PROMPT) == (FULL, "indent")


def test_free_format_program_is_left_alone():
    free = "IDENTIFICATION DIVISION.\nPROGRAM-ID. T.\nPROCEDURE DIVISION.\nGOBACK.\n"
    assert repair(free, PROMPT) == (free, None)


def test_answer_without_program_id_gets_the_program_start_from_the_prompt():
    # the answer repeats the last prompt line (WORKING-STORAGE SECTION.): it is not doubled
    assert repair("       WORKING-STORAGE SECTION.\n" + BODY, PROMPT) == (FULL, "prefix")
    assert repair(BODY, PROMPT) == (FULL, "prefix")                 # answer starts after it: whole start goes in


def test_none_stays_none():
    assert repair(None, PROMPT) == (None, None)
