"""Exact dedup over all sources together. Earlier source in PRIORITY wins, so CodeNet is
deduplicated against Stack v1 and X-COBOL, as the codenet manifest plans.

    rows = exact_dedup([*read_manifest(a), *read_manifest(b), *read_manifest(c)])

Near-dups: MinHash + LSH over kept rows; one file per cluster is kept.

    python -m onprem_codegen.data.dedup   # data/raw/raw-corpus/manifest_filtered.parquet -> manifest_dedup.parquet
"""
import collections
import json
from dataclasses import replace
from pathlib import Path

import yaml
from datasketch import MinHash, MinHashLSH

from onprem_codegen.data.schema import ManifestRow, read_manifest, write_manifest

PRIORITY = {"stack_v1": 0, "xcobol": 1, "codenet": 2}


def exact_dedup(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Same sha256 => same `id`, so the keeper is referenced as `source:path` in exact_dup_of."""
    first: dict[str, ManifestRow] = {}
    for r in sorted((r for r in rows if r.kept), key=lambda r: (PRIORITY[r.source], r.path)):
        keeper = first.setdefault(r.sha256, r)
        if keeper is not r:
            r.exact_dup_of, r.kept, r.filter_reason = f"{keeper.source}:{keeper.path}", False, "exact_dup"
    return rows


def near_dedup(rows: list[ManifestRow], contents: dict[str, str], cfg: dict) -> list[ManifestRow]:
    """Greedy: rows in PRIORITY order; a row similar to an earlier kept one joins its cluster and is dropped."""
    lsh = MinHashLSH(threshold=cfg["threshold"], num_perm=cfg["num_perm"])
    cluster: dict[str, int] = {}  # row id -> cluster id of the kept row
    out = {}
    for r in sorted(rows, key=lambda r: (PRIORITY[r.source], r.id)):
        if not r.kept:
            out[id(r)] = r
            continue
        toks = contents[r.id].upper().split()
        m = MinHash(num_perm=cfg["num_perm"])
        for i in range(max(len(toks) - cfg["shingle"] + 1, 1)):
            m.update(" ".join(toks[i:i + cfg["shingle"]]).encode())
        hit = lsh.query(m)
        if hit:
            out[id(r)] = replace(r, kept=False, filter_reason="near_dup", dup_cluster_id=cluster[hit[0]])
        else:
            cluster[r.id] = len(cluster)
            lsh.insert(r.id, m)
            out[id(r)] = replace(r, dup_cluster_id=cluster[r.id])
    return [out[id(r)] for r in rows]


def main() -> None:
    raw = Path("data/raw/raw-corpus")
    cfg = yaml.safe_load(Path("configs/data/dedup.yaml").read_text())
    contents = {}
    for p in raw.glob("*/corpus.jsonl"):
        for line in open(p):
            d = json.loads(line)
            contents[d["id"]] = d["content"]
    rows = exact_dedup(read_manifest(str(raw / "manifest_filtered.parquet")))  # in place; keeps order
    rows = near_dedup(rows, contents, cfg)
    write_manifest(rows, str(raw / "manifest_dedup.parquet"))
    print(collections.Counter(r.filter_reason or "kept" for r in rows))


if __name__ == "__main__":
    main()
