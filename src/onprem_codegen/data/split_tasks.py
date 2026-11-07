"""Held-out task split for sft-v1: a random fraction of task-pool, chosen before any new model result exists.

    python -m onprem_codegen.data.split_tasks [--task-pool PATH] [--seed 7] [--fraction 0.2] [--out PATH]

Reads task-pool:v1 (W&B artifact, or a local task_pool.jsonl) and writes data/splits/heldout_task_ids.json.
These tasks must not enter rejection sampling or training; the file is committed so the choice is fixed.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from onprem_codegen.utils import read_jsonl

ARTIFACT = "task-pool:v1"
OUT = Path("data/splits/heldout_task_ids.json")


def heldout_ids(task_ids: list[str], fraction: float, seed: int) -> list[str]:
    """Sorted ids of round(n * fraction) tasks; independent of input order and duplicates."""
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    ids = sorted(set(task_ids))
    random.Random(seed).shuffle(ids)
    return sorted(ids[:round(len(ids) * fraction)])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-pool", help=f"local task_pool.jsonl instead of the W&B artifact {ARTIFACT}")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    if a.task_pool:
        pool = Path(a.task_pool)
    else:
        import wandb
        path = f"{os.environ['WANDB_ENTITY']}/{os.environ.get('WANDB_PROJECT', 'onprem-codegen')}/{ARTIFACT}"
        pool = Path(wandb.Api().artifact(path).download()) / "task_pool.jsonl"

    ids = [t["task_id"] for t in read_jsonl(pool)]
    held = heldout_ids(ids, a.fraction, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": a.seed, "fraction": a.fraction, "task_pool": ARTIFACT,
        "task_pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        "n_tasks": len(set(ids)), "task_ids": held}, indent=1) + "\n")
    print(f"held out {len(held)} of {len(set(ids))} tasks -> {out}")


if __name__ == "__main__":
    main()
