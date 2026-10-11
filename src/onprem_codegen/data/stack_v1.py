"""Stack v1 (dedup) COBOL subset -> raw-corpus records + manifest rows (source=stack_v1).

Writes `<out_dir>/corpus.jsonl` (contracts.md record schema) and returns the manifest rows.
`cfg` is configs/data/stack_v1_cobol.yaml. The dataset is gated: accept the terms on HF and set HF_TOKEN first.
"""
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

from onprem_codegen.data.schema import ManifestRow


def load_stack_v1(cfg, out_dir: str | Path, ds=None) -> list[ManifestRow]:
    ds = ds if ds is not None else load_dataset(
        cfg["hf_repo"], data_dir=cfg["data_dir"], revision=cfg["revision"], split="train")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "corpus.jsonl", "w") as f:
        for x in ds:
            b = x["content"].encode()
            h = hashlib.sha256(b).hexdigest()
            lic = (x.get("max_stars_repo_licenses") or [None])[0]  # first listed; per-file license
            f.write(json.dumps({
                "id": h, "source": "stack_v1_cobol", "content": x["content"], "content_sha256": h,
                "license": lic,
                "meta": {"original_path": x["max_stars_repo_path"], "original_license": lic},
                "syntax_check": {"status": "skipped", "stderr": ""}}) + "\n")
            rows.append(ManifestRow(
                id=h, source="stack_v1", source_version=cfg["revision"], path=x["max_stars_repo_path"],
                ext=x.get("ext") and "." + x["ext"].lower().lstrip(".") or Path(x["max_stars_repo_path"]).suffix.lower(),
                repo=x["max_stars_repo_name"], sha256=h, license=lic, license_ok=lic is not None,
                n_bytes=len(b), filter_reason=None, syntax_check="skipped", syntax_error=None,
                exact_dup_of=None, dup_cluster_id=None, kept=True, decontam_hit=False))
    return rows
