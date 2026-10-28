"""CodeNet loader: accepted COBOL submissions -> raw-corpus records + manifest rows (source=codenet_cobol).

Consumes the tree extracted by download_codenet.py, not the download itself:
    <root>/metadata/<problem_id>.csv, <root>/data/<problem_id>/COBOL/<submission_id>.<ext>
Writes `<out_dir>/corpus.jsonl` (contracts.md record schema) and returns the manifest rows.
"""
import csv
import json
from pathlib import Path

from onprem_codegen.data.schema import ManifestRow, raw_record


def load_codenet(cfg, root: str | Path, out_dir: str | Path) -> list[ManifestRow]:
    """`cfg` is configs/data/codenet_cobol.yaml."""
    root, out, version, license = Path(root), Path(out_dir), cfg["version"], cfg["license"]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "corpus.jsonl", "w") as f:
        for meta in sorted((root / "metadata").glob("p[0-9]*.csv")):
            with open(meta, newline="") as m:
                accepted = {r["submission_id"]: r["filename_ext"] for r in csv.DictReader(m)
                            if r["language"] == "COBOL" and r["status"] == "Accepted"}
            for sid, ext in sorted(accepted.items()):
                p = root / "data" / meta.stem / "COBOL" / f"{sid}.{ext}"
                if not p.exists():
                    continue
                content = p.read_bytes().decode("utf-8", errors="replace")
                rec, h, n = raw_record("codenet_cobol", content, license, {"problem_id": meta.stem, "submission_id": sid})
                f.write(json.dumps(rec) + "\n")
                rows.append(ManifestRow(
                    id=h, source="codenet_cobol", source_version=version,
                    path=str(p.relative_to(root)), ext=p.suffix.lower(), repo=None, sha256=h, license=license,
                    license_ok=license is not None,  # license is only tagged, not filtered (see licenses.yaml)
                    n_bytes=n))
    return rows
