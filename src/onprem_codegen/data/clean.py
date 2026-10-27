"""Step 2: raw-corpus -> filters -> exact + near-dup dedup -> clean-corpus artifact.

    python -m onprem_codegen.data.clean [--no-wandb]

Output: data/raw/clean-corpus/{manifest.parquet (all rows), corpus.jsonl (kept only), stats.json}
ponytail: reads raw-corpus from local data/raw/raw-corpus (only registers it as run input for lineage);
download the artifact instead when running on a fresh instance.
Syntax check (cobc -fsyntax-only in the sandbox) only tags kept files; it never drops them.
"""
import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from onprem_codegen.data.decontaminate import decontaminate
from onprem_codegen.data.dedup import exact_dedup, near_dedup
from onprem_codegen.data.filters import apply_filters
from onprem_codegen.data.schema import read_manifest, write_manifest
from onprem_codegen.eval.execute import execute
from onprem_codegen.utils import git_commit

RAW, OUT = Path("data/raw/raw-corpus"), Path("data/raw/clean-corpus")


def syntax_check(rows, contents, records, timeout_s=30.0, workers=8):
    """Tags kept non-.cpy rows (and their corpus records) with the cobc status; copybooks stay "skipped"."""
    ids = list(dict.fromkeys(r.id for r in rows if r.kept and r.ext != ".cpy"))
    with ThreadPoolExecutor(workers) as pool:
        results = dict(zip(ids, pool.map(lambda i: execute(contents[i], [], timeout_s=timeout_s), ids)))
    for i, res in results.items():
        records[i]["syntax_check"] = {"status": res.status.value, "stderr": res.compile_stderr[:500]}
    for r in rows:
        if r.id in results and r.kept:
            sc = records[r.id]["syntax_check"]
            r.syntax_check, r.syntax_error = sc["status"], sc["stderr"] or None
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    cfg = {n: yaml.safe_load(Path(f"configs/data/{n}.yaml").read_text()) for n in ("filters", "decontam", "dedup")}

    records = {}
    for p in RAW.glob("*/corpus.jsonl"):
        for line in open(p):
            d = json.loads(line)
            records[d["id"]] = d
    contents = {i: d["content"] for i, d in records.items()}

    rows = apply_filters(read_manifest(str(RAW / "manifest.parquet")), contents, cfg["filters"])
    rows, report = decontaminate(rows, contents, cfg["decontam"])
    rows = near_dedup(exact_dedup(rows), contents, cfg["dedup"])
    rows = syntax_check(rows, contents, records)

    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, str(OUT / "manifest.parquet"))
    kept = [r for r in rows if r.kept]
    with open(OUT / "corpus.jsonl", "w") as f:
        for i in dict.fromkeys(r.id for r in kept):  # one record per content hash
            f.write(json.dumps(records[i]) + "\n")

    (OUT / "decontam_report.json").write_text(json.dumps(report, indent=1))
    stats = {"files_in": len(rows), "files_kept": len(kept), "bytes_kept": sum(r.n_bytes for r in kept),
             "by_reason": dict(collections.Counter(r.filter_reason or "kept" for r in rows))}
    stats["by_syntax_check"] = dict(collections.Counter(r.syntax_check for r in kept))
    for key in ("source", "ext", "license"):
        stats[f"kept_by_{key}"] = dict(collections.Counter(str(getattr(r, key)) for r in kept).most_common())
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    if not a.no_wandb:
        import wandb
        commit = git_commit() or ""
        with wandb.init(project="onprem-codegen", job_type="clean", name=f"clean-{commit[:7]}",
                        config={**cfg, "git_commit": commit}) as run:
            run.use_artifact("raw-corpus:latest")
            run.summary.update({k: v for k, v in stats.items() if not isinstance(v, dict)})
            run.summary["by_syntax_check"] = stats["by_syntax_check"]
            art = wandb.Artifact("clean-corpus", type="dataset", metadata=stats)
            art.add_dir(str(OUT))
            run.log_artifact(art)
            rep = wandb.Artifact("decontam-report", type="report", metadata={"hits": len(report)})
            rep.add_file(str(OUT / "decontam_report.json"))
            run.log_artifact(rep)


if __name__ == "__main__":
    main()
