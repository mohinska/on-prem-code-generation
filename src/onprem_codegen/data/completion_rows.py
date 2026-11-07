"""sft-dataset rows -> about half of the tasks as completion rows (task 1.5), the rest stay full programs.

A completion row: prompt = task text + program up to DATA DIVISION. + IO_COMPLETE, completion = rest of the program.

    python -m onprem_codegen.data.completion_rows [--dataset sft-dataset:v1-lite | --input sft.jsonl] [--seed 7] [--no-wandb]

Programs are not changed (prefix + answer = the checked program), so no new execution is needed. The prompts are new,
so the benchmark check runs again; with no hit the W&B artifact gets the aliases v1-lite and decontaminated-v1.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import yaml

from onprem_codegen.data.decontaminate import run as decontaminate
from onprem_codegen.prompt import IO_COMPLETE, IO_STDIN, build_prompt, instruction
from onprem_codegen.utils import git_commit, read_jsonl

OUT = Path("data/raw/sft-dataset-v1-lite-completion")

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sft-dataset:v1-lite", help="W&B sft-dataset version (GOBACK rows, task 1.2)")
    ap.add_argument("--input", help="local sft.jsonl instead of the W&B artifact")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    run = None
    if not a.no_wandb:
        from onprem_codegen.utils import init_run
        run = init_run("sft-v1-lite-completion", {"dataset": a.dataset, "seed": a.seed})
    src = Path(a.input) if a.input else Path(run.use_artifact(a.dataset).download()) / "sft.jsonl"
    rows = make_completion_rows(read_jsonl(src), a.seed)
    (out / "sft.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n_comp = sum((r.get("meta") or {}).get("format") == "completion" for r in rows)
    cfg = yaml.safe_load(Path("configs/data/decontam.yaml").read_text())
    rc = decontaminate(out / "sft.jsonl", out / "decontam_report.json", cfg)
    stats = {"rows": len(rows), "tasks": len({r["task_id"] for r in rows}), "seed": a.seed, "completion_rows": n_comp,
             "completion_share": round(n_comp / max(len(rows), 1), 4), "decontam_hits": rc}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)
    if run:
        import wandb
        art = wandb.Artifact("sft-dataset", type="dataset", metadata={**stats, "git_commit": git_commit()})
        for f in ("sft.jsonl", "stats.json", "decontam_report.json"):
            art.add_file(str(out / f))
        run.log_artifact(art, aliases=["v1-lite-completion"] + (["v1-lite", "decontaminated-v1"] if rc == 0 else []))
        run.finish()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
