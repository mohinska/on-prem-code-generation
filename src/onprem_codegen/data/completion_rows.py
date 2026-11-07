"""sft-dataset rows -> about half of the tasks as completion rows (task 1.5), the rest stay full programs.

A completion row: prompt = task text + program up to DATA DIVISION. + IO_COMPLETE, completion = rest of the program.
Library only; the CLI / artifact step is wired separately.
"""
import hashlib
import random
import re

from onprem_codegen.prompt import IO_COMPLETE, IO_STDIN, build_prompt, instruction

_DATA_DIV = re.compile(r"^[ \t]*DATA[ \t]+DIVISION\.", re.M | re.I)  # a comment line has '*' first, so no match
_FENCE = re.compile(r"\A```cobol\n(.*)```\Z", re.S)


def _task_text(prompt: str) -> str:
    """Inverse of build_prompt(task_text) for the default stdin instruction."""
    tail = f"\n\n{instruction(IO_STDIN)}"
    if not prompt.endswith(tail):
        raise ValueError(f"prompt does not end with the standard instruction: {prompt[-80:]!r}")
    return prompt[: -len(tail)]


def _split(program: str) -> tuple[str, str] | None:
    m = _DATA_DIV.search(program)
    if not m:
        return None
    cut = len(program[: m.start()].rstrip()) + 1  # prefix ends in one "\n"; blank lines before DATA DIVISION go to the answer
    return program[:cut], program[cut:]


def make_completion_rows(rows: list[dict], seed: int, fraction: float = 0.5) -> list[dict]:
    """Same order, new dicts for converted rows. Task choice: seeded shuffle of sorted task_ids, so not per row."""
    ids = sorted({r["task_id"] for r in rows})
    random.Random(seed).shuffle(ids)
    chosen = set(ids[: round(len(ids) * fraction)])
    out = []
    for r in rows:
        m = _FENCE.match(r["completion"])
        parts = _split(m.group(1)) if m and r["task_id"] in chosen else None
        if parts is None:
            out.append(r)
            continue
        prefix, rest = parts
        completion = f"```cobol\n{rest}```"
        out.append({**r, "prompt": build_prompt(f"{_task_text(r['prompt'])}\n\n{prefix}", IO_COMPLETE),
                    "completion": completion, "content_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                    "meta": {**(r.get("meta") or {}), "format": "completion"}})
    return out
