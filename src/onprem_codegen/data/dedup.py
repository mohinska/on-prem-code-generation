"""Exact dedup over all sources together. Earlier source in PRIORITY wins, so CodeNet is
deduplicated against Stack v1 and X-COBOL, as the codenet manifest plans.

    rows = exact_dedup([*read_manifest(a), *read_manifest(b), *read_manifest(c)])

ponytail: exact only; MinHash near-dups (dup_cluster_id) plug in here next.
"""
from onprem_codegen.data.schema import ManifestRow

PRIORITY = {"stack_v1": 0, "xcobol": 1, "codenet": 2}


def exact_dedup(rows: list[ManifestRow]) -> list[ManifestRow]:
    first: dict[str, str] = {}
    for r in sorted(rows, key=lambda r: PRIORITY[r.source]):
        keeper = first.setdefault(r.sha256, r.id)
        if keeper != r.id:
            r.exact_dup_of, r.kept, r.filter_reason = keeper, False, "exact_dup"
    return rows
