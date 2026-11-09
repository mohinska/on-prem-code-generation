"""Free-format COBOL -> fixed format, so every SFT program starts its code in column 8.

Human CodeNet programs are often free format (code from column 1). `cobc -x` detects that ("free format detected"),
so they pass their tests, but COBOLEval compiles with -fformat=variable, where column 7 is the indicator. Trained on a
mix of both, sft-v0 and sft-v1 wrote the first line from column 1 and the rest indented: `invalid indicator 'F' at
column 7`. A shifted line can pass column 72 and lose its tail, so the caller re-runs the tests on the result.
"""
import re

_FIXED_LINE = re.compile(r"\A[ 0-9]{6}(?:[ *\-/dD$].*)?\Z")   # sequence area, then indicator (or nothing)


def is_free(program: str) -> bool:
    """True if any non-blank line does not fit the fixed layout (6-char sequence area + indicator)."""
    return any(line.strip() and not _FIXED_LINE.match(line) for line in program.splitlines())


def to_fixed(program: str) -> str:
    """Free format: every non-blank line indented by 7 spaces (column 8, area A). Fixed programs are returned as they are."""
    if not is_free(program):
        return program
    return "".join(("       " + line if line.strip() else line) for line in program.splitlines(keepends=True))
