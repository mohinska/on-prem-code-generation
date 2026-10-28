"""Per-file corpus manifest: one row per file, stored as Parquet next to the corpus
and logged inside the same W&B artifact (`manifest.parquet`)."""
import hashlib
from dataclasses import asdict, dataclass, fields

import pyarrow as pa
import pyarrow.parquet as pq

SOURCES = ("stack_v1_cobol", "xcobol", "codenet_cobol")  # same names as `source` in corpus.jsonl records


@dataclass
class ManifestRow:
    id: str                          # content sha256 for raw rows; stable per file
    source: str                      # one of SOURCES
    source_version: str              # pinned revision / commit / release
    path: str                        # original path in the source
    ext: str                         # file extension, lowercase, with dot (e.g. .cbl)
    repo: str | None                 # None for sources without repos
    sha256: str                      # of file content
    license: str | None              # SPDX id
    license_ok: bool
    n_bytes: int
    filter_reason: str | None = None  # why dropped (None if not dropped by a filter)
    syntax_check: str = "skipped"    # ok | compile_error | timeout | skipped (execute status)
    syntax_error: str | None = None  # short stderr, <= 500 chars
    exact_dup_of: str | None = None  # id of the kept exact duplicate
    dup_cluster_id: int | None = None  # MinHash near-dup cluster
    kept: bool = True                # final: in clean-corpus
    decontam_hit: bool = False       # matched COBOLEval


def raw_record(source: str, content: str, license: str | None, meta: dict) -> tuple[dict, str, int]:
    """corpus.jsonl record (contracts.md schema), plus the content sha256 and byte size for the manifest row."""
    b = content.encode()
    h = hashlib.sha256(b).hexdigest()
    return ({"id": h, "source": source, "content": content, "content_sha256": h, "license": license, "meta": meta,
             "syntax_check": {"status": "skipped", "stderr": ""}}, h, len(b))


def write_manifest(rows: list[ManifestRow], path: str) -> None:
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), path)


def read_manifest(path: str) -> list[ManifestRow]:
    names = [f.name for f in fields(ManifestRow)]
    return [ManifestRow(**r) for r in pq.read_table(path, columns=names).to_pylist()]
