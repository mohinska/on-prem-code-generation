"""Step 1: collect Stack v1 + X-COBOL (+ CodeNet if present) -> raw-corpus artifact.

    python -m onprem_codegen.data.collect [--xcobol-root DIR] [--codenet-root DIR] [--skip SOURCE ...] [--no-wandb]

A source whose data is missing is an error; pass --skip <source> to leave it out on purpose.

Output: data/raw/raw-corpus/{manifest.parquet, {stack_v1,xcobol,codenet}/corpus.jsonl, stats.json}
"""
import argparse
import collections
import json
from dataclasses import replace
from pathlib import Path

import yaml

from onprem_codegen.data.codenet import load_codenet
from onprem_codegen.data.schema import SOURCES, write_manifest
from onprem_codegen.data.stack_v1 import load_stack_v1
from onprem_codegen.data.xcobol import load_xcobol
from onprem_codegen.utils import init_run

CFG = Path("configs/data")
OUT = Path("data/raw/raw-corpus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xcobol-root", default="data/raw/xcobol")
    ap.add_argument("--codenet-root", default="data/raw/codenet_cobol/Project_CodeNet")
    ap.add_argument("--skip", action="append", default=[], choices=SOURCES)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    cfg = {n: yaml.safe_load((CFG / f"{n}.yaml").read_text()) for n in ("stack_v1_cobol", "xcobol", "codenet_cobol", "licenses")}

    loaders = {  # source -> (data root that must exist, loader)
        "stack_v1_cobol": (None, lambda: load_stack_v1(cfg["stack_v1_cobol"], OUT / "stack_v1")),
        "xcobol": (a.xcobol_root, lambda: load_xcobol(cfg["xcobol"], a.xcobol_root, OUT / "xcobol")),
        "codenet_cobol": (a.codenet_root, lambda: load_codenet(cfg["codenet_cobol"], a.codenet_root, OUT / "codenet")),
    }
    rows = []
    for name, (root, load) in loaders.items():
        if name in a.skip:
            continue
        if root and not Path(root).exists():
            raise SystemExit(f"{name}: no data at {root} (fetch it, or pass --skip {name})")
        rows += load()
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
        with init_run("collect", cfg) as run:
            run.summary.update({k: v for k, v in stats.items() if not isinstance(v, dict)})
            art = wandb.Artifact("raw-corpus", type="dataset", metadata=stats)
            art.add_dir(str(OUT))
            run.log_artifact(art)


if __name__ == "__main__":
    main()
