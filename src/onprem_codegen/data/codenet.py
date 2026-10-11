"""CodeNet loader: accepted COBOL submissions -> manifest rows (source=codenet).

Consumes an extracted Project_CodeNet tree (the download artifact), not the download itself:
    <root>/metadata/<problem_id>.csv, <root>/data/<problem_id>/COBOL/<submission_id>.cbl
"""
import csv
import hashlib
from pathlib import Path

from onprem_codegen.data.schema import ManifestRow


def load_codenet(root: str | Path, version: str, license: str | None = None) -> list[ManifestRow]:
    root = Path(root)
    rows = []
    for meta in sorted((root / "metadata").glob("*.csv")):
        with open(meta, newline="") as f:
            accepted = {r["submission_id"] for r in csv.DictReader(f)
                        if r["language"] == "COBOL" and r["status"] == "Accepted"}
        for sid in sorted(accepted):
            p = root / "data" / meta.stem / "COBOL" / f"{sid}.cbl"
            if not p.exists():
                continue
            b = p.read_bytes()
            h = hashlib.sha256(b).hexdigest()
            rows.append(ManifestRow(
                id=f"codenet/{meta.stem}/{sid}", source="codenet", source_version=version,
                path=str(p.relative_to(root)), ext=p.suffix.lower(), repo=None, sha256=h, license=license,
                license_ok=license is not None,  # ponytail: real license filter lives in clean stage
                n_bytes=len(b), filter_reason=None, syntax_check="skipped", syntax_error=None,
                exact_dup_of=None, dup_cluster_id=None, kept=True, decontam_hit=False))
    return rows
