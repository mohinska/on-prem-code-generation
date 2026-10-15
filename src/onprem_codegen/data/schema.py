"""Per-file corpus manifest: one row per file, stored as Parquet next to the corpus
and logged inside the same W&B artifact (`manifest.parquet`)."""
from dataclasses import asdict, dataclass, fields

import pyarrow as pa
import pyarrow.parquet as pq

SOURCES = ("stack_v1", "xcobol", "codenet")


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
    filter_reason: str | None        # why dropped (None if not dropped by a filter)
    syntax_check: str                # ok | compile_error | timeout | skipped (run_program status)
    syntax_error: str | None         # short stderr, <= 500 chars
    exact_dup_of: str | None         # id of the kept exact duplicate
    dup_cluster_id: int | None       # MinHash near-dup cluster
    kept: bool                       # final: in clean-corpus
    decontam_hit: bool               # matched COBOLEval


def write_manifest(rows: list[ManifestRow], path: str) -> None:
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), path)


def read_manifest(path: str) -> list[ManifestRow]:
    names = [f.name for f in fields(ManifestRow)]
    return [ManifestRow(**r) for r in pq.read_table(path, columns=names).to_pylist()]
