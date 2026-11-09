from onprem_codegen.data.fixed_format import is_free, to_fixed

FIXED = "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. T.\n      * comment\n       PROCEDURE DIVISION.\n           GOBACK.\n"
FREE = "IDENTIFICATION DIVISION.\nPROGRAM-ID. T.\n*> comment\nPROCEDURE DIVISION.\n  DISPLAY 'HI'.\n  GOBACK.\n"
MIXED = "IDENTIFICATION DIVISION.\n       PROGRAM-ID. T.\n       PROCEDURE DIVISION.\n           GOBACK.\n"


def test_fixed_program_is_not_free_and_stays_byte_identical():
    assert not is_free(FIXED)
    assert to_fixed(FIXED) == FIXED


def test_free_and_mixed_programs_are_free():
    assert is_free(FREE) and is_free(MIXED)


def test_to_fixed_indents_every_non_blank_line_by_seven():
    out = to_fixed(FREE)
    assert not is_free(out)
    assert out.splitlines()[0] == "       IDENTIFICATION DIVISION."
    assert out.splitlines()[4] == "         DISPLAY 'HI'."
    assert to_fixed("A.\n\nB.\n") == "       A.\n\n       B.\n"   # blank lines stay blank


def test_sequence_numbers_and_indicators_count_as_fixed():
    assert not is_free("000100 IDENTIFICATION DIVISION.\n000200*COMMENT\n      -    'CONT'.\n")
