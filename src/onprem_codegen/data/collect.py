"""Step 1: collect Stack v1 + X-COBOL (+ CodeNet if present) -> raw-corpus artifact.

    python -m onprem_codegen.data.collect [--codenet-root DIR] [--no-wandb]

Output: data/raw/raw-corpus/{manifest.parquet, stack_v1/corpus.jsonl, xcobol/corpus.jsonl, stats.json}
"""
import argparse
import collections
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml

from onprem_codegen.data.codenet import load_codenet
from onprem_codegen.data.schema import write_manifest
from onprem_codegen.data.stack_v1 import load_stack_v1
from onprem_codegen.data.xcobol import load_xcobol

CFG = Path("configs/data")
OUT = Path("data/raw/raw-corpus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xcobol-root", default="data/raw/xcobol")
    ap.add_argument("--codenet-root")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    cfg = {n: yaml.safe_load((CFG / f"{n}.yaml").read_text()) for n in ("stack_v1_cobol", "xcobol", "licenses")}

    rows = load_stack_v1(cfg["stack_v1_cobol"], OUT / "stack_v1") + load_xcobol(cfg["xcobol"], a.xcobol_root, OUT / "xcobol")
    if a.codenet_root:
        rows += load_codenet(a.codenet_root, version="release")
    allow = set(cfg["licenses"]["allow"])
    rows = [replace(r, license_ok=r.license is None or r.license in allow) for r in rows]
    write_manifest(rows, str(OUT / "manifest.parquet"))

    stats = {"files": len(rows), "bytes": sum(r.n_bytes for r in rows)}
    for key in ("source", "ext", "license"):
        stats[f"by_{key}"] = dict(collections.Counter(str(getattr(r, key)) for r in rows).most_common())
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    if not a.no_wandb:
        import wandb
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        with wandb.init(project="onprem-codegen", job_type="collect", name=f"collect-{commit[:7]}", config={**cfg, "git_commit": commit}) as run:
            run.summary.update({k: v for k, v in stats.items() if not isinstance(v, dict)})
            art = wandb.Artifact("raw-corpus", type="dataset", metadata=stats)
            art.add_dir(str(OUT))
            run.log_artifact(art)


if __name__ == "__main__":
    main()
