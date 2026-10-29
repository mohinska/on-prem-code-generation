"""Step 2: raw-corpus -> filters -> exact + near-dup dedup -> clean-corpus artifact.

    python -m onprem_codegen.data.clean [--no-wandb]

Output: data/raw/clean-corpus/{manifest.parquet (all rows), corpus.jsonl (kept only), stats.json}
Reads raw-corpus from --raw-dir (default data/raw/raw-corpus); downloads the W&B artifact when it is not there.
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
from onprem_codegen.utils import init_run

RAW, OUT = Path("data/raw/raw-corpus"), Path("data/raw/clean-corpus")


def syntax_check(rows, contents, records, timeout_s=30.0, workers=8):
    """Tags kept non-.cpy rows (and their corpus records) with the cobc status; copybooks stay "skipped"."""
    ids = list(dict.fromkeys(r.id for r in rows if r.kept and r.ext != ".cpy"))
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda i: execute(contents[i], [], timeout_s=timeout_s), ids))
    sc = {i: {"status": res.status.value, "stderr": res.compile_stderr[:500]} for i, res in zip(ids, results)}
    for i, v in sc.items():
        records[i]["syntax_check"] = v
    for r in rows:
        if r.kept and r.id in sc:
            r.syntax_check, r.syntax_error = sc[r.id]["status"], sc[r.id]["stderr"] or None
    return rows


def ensure_raw(run, raw: Path) -> None:
    """Registers raw-corpus:latest as run input (lineage); downloads it when there is no local copy."""
    art = run.use_artifact("raw-corpus:latest")
    if not (raw / "manifest.parquet").exists():
        art.download(root=str(raw))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    raw = a.raw_dir or RAW
    cfg = {n: yaml.safe_load(Path(f"configs/data/{n}.yaml").read_text()) for n in ("filters", "decontam", "dedup")}

    run = None if a.no_wandb else init_run("clean", cfg)
    if run:
        ensure_raw(run, raw)

    records = {}
    for p in raw.glob("*/corpus.jsonl"):
        for line in open(p):
            d = json.loads(line)
            records[d["id"]] = d
    contents = {i: d["content"] for i, d in records.items()}

    rows = apply_filters(read_manifest(str(raw / "manifest.parquet")), contents, cfg["filters"])
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
    stats["by_benchmark"] = dict(collections.Counter(e["benchmark"] for e in report))
    stats["by_syntax_check"] = dict(collections.Counter(r.syntax_check for r in kept))
    for key in ("source", "ext", "license", "license_ok"):
        stats[f"kept_by_{key}"] = dict(collections.Counter(str(getattr(r, key)) for r in kept).most_common())
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    if run:
        import wandb
        run.summary.update(stats)
        art = wandb.Artifact("clean-corpus", type="dataset", metadata=stats)
        art.add_dir(str(OUT))
        run.log_artifact(art)
        rep = wandb.Artifact("decontam-report", type="report", metadata={"hits": len(report)})
        rep.add_file(str(OUT / "decontam_report.json"))
        run.log_artifact(rep)
        run.finish()


if __name__ == "__main__":
    main()
