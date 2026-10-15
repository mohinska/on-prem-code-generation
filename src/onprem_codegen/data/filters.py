"""Size/text filters over raw-corpus. Sets kept=False and filter_reason; never deletes rows.

    python -m onprem_codegen.data.filters   # reads data/raw/raw-corpus, writes manifest_filtered.parquet
"""
import json
from dataclasses import replace
from pathlib import Path

import yaml

from onprem_codegen.data.schema import ManifestRow, read_manifest, write_manifest

RAW = Path("data/raw/raw-corpus")


def reason(content: str, cfg: dict) -> str | None:
    if "\x00" in content or "�" in content:
        return "not_text"
    n = content.count("\n") + 1
    if n < cfg["min_lines"]:
        return "too_short"
    if n > cfg["max_lines"]:
        return "too_long"
    s = [c for c in content if not c.isspace()]
    if s and sum(not c.isalnum() for c in s) / len(s) > cfg["max_nonalnum_ratio"]:
        return "high_nonalnum"
    return None


def apply_filters(rows: list[ManifestRow], contents: dict[str, str], cfg: dict) -> list[ManifestRow]:
    out = []
    for r in rows:
        why = r.filter_reason or reason(contents[r.id], cfg)
        out.append(replace(r, kept=r.kept and why is None, filter_reason=why))
    return out


def main() -> None:
    cfg = yaml.safe_load(Path("configs/data/filters.yaml").read_text())
    contents = {}
    for p in RAW.glob("*/corpus.jsonl"):
        for line in open(p):
            d = json.loads(line)
            contents[d["id"]] = d["content"]
    rows = apply_filters(read_manifest(str(RAW / "manifest.parquet")), contents, cfg)
    write_manifest(rows, str(RAW / "manifest_filtered.parquet"))
    import collections
    print(collections.Counter(r.filter_reason or "kept" for r in rows))


if __name__ == "__main__":
    main()
