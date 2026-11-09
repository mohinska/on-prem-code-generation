"""Diagnostic only: repair the two sft-v1 format defects in stored answers and execute them again.

    python -m onprem_codegen.eval.repair_format IN_results.jsonl OUT_generations.jsonl

It edits model answers, so its numbers are never a result of a model (see docs/sft-v1.md). Applied the same way to
base and adapted answers. Two rules, both from the sft-v1 diagnosis:
- first line from column 1 and every other line in fixed layout -> the first line gets its 7-space indent;
- no PROGRAM-ID -> the program start from the prompt goes in front (the answer is only the rest of the program).
Only COBOLEval and CobolCodeBench completion rows are written (the groups where the prompt holds the program start).
"""
import json
import re
import sys

from onprem_codegen.data.fixed_format import _FIXED_LINE
from onprem_codegen.utils import read_jsonl

_INSTRUCTION = "\n\nWrite a complete GnuCOBOL program"
_PROGRAM_ID = re.compile(r"PROGRAM-ID", re.I)


def _indent_first(program: str) -> str | None:
    lines = program.splitlines(keepends=True)
    code = [i for i, l in enumerate(lines) if l.strip()]
    if not code or _FIXED_LINE.match(lines[code[0]].rstrip("\n")):
        return None
    if all(_FIXED_LINE.match(lines[i].rstrip("\n")) for i in code[1:]):
        lines[code[0]] = "       " + lines[code[0]]
        return "".join(lines)
    return None   # free format throughout: not this defect


def repair(program: str | None, prompt: str) -> tuple[str | None, str | None]:
    """(program, rule applied or None)."""
    if program is None:
        return None, None
    if not _PROGRAM_ID.search(program):
        start = prompt.split(_INSTRUCTION)[0].rstrip("\n").splitlines()
        first = next((l.strip() for l in program.splitlines() if l.strip()), "")
        cut = max((i for i, l in enumerate(start) if l.strip() == first), default=len(start))   # don't repeat it
        return "\n".join(start[:cut]) + "\n" + program, "prefix"
    fixed = _indent_first(program)
    return (fixed, "indent") if fixed else (program, None)


def main(src: str, dst: str) -> None:
    n, rules = 0, {"indent": 0, "prefix": 0, None: 0}
    with open(dst, "w", encoding="utf-8") as f:
        for r in read_jsonl(src):
            if not (r["benchmark"] == "coboleval" or (r["benchmark"] == "cobolcodebench" and r["task_mode"] == "completion")):
                continue
            program, rule = repair(r.get("program"), r["prompt"])
            rules[rule] += 1
            n += 1
            gen = r["generation"] if program is None else f"```cobol\n{program}```"
            keep = ("task_id", "benchmark", "task_mode", "mode", "decoding", "run", "sample_idx", "prompt",
                    "reasoning", "finish_reason", "completion_tokens", "served_model", "seed")
            f.write(json.dumps({**{k: r[k] for k in keep}, "generation": gen, "repair": rule}, ensure_ascii=False) + "\n")
    print(f"{n} rows, repaired: indent {rules['indent']}, prefix {rules['prefix']}, unchanged {rules[None]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
