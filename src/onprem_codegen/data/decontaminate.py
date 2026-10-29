"""Flag corpus files / SFT rows that overlap benchmark tasks, by n-gram containment.

    python -m onprem_codegen.data.decontaminate --input <jsonl> --report <path>
    python -m onprem_codegen.data.decontaminate --artifact sft-dataset:latest   # W&B: check, then add the alias

Every part of every task (prompt, canonical solution, caller programs, expected outputs) of every benchmark in
configs/data/decontam.yaml is shingled; a text is a hit when it contains >= `threshold` of the shingles of one part.
Exit code 1 if there is a hit, so a pipeline can gate on it.
"""
import argparse
import collections
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from onprem_codegen.data.schema import ManifestRow
from onprem_codegen.eval.benchmarks import load


def _norm(text: str) -> list[str]:
    lines = [re.sub(r"\*>.*", "", l) for l in text.splitlines() if not (len(l) > 6 and l[6] in "*/")]
    return " ".join(lines).upper().split()


def _shingles(toks: list[str], n: int) -> set[int]:
    return {hash(" ".join(toks[i:i + n])) for i in range(max(len(toks) - n + 1, 1))}


def _parts(t):
    """(name, text) of everything in a task that must not leak into training data."""
    yield "prompt", t.prompt
    yield "canonical_solution", t.canonical_solution
    for x in t.tests:
        yield "caller", x.get("caller")
        yield "expected", x.get("expected")
        for v in x.get("expected_outputs", {}).values():
            yield "expected_output", v


def _index(cfg: dict):
    """shingle -> keys of the task parts containing it, and the shingle count of each part."""
    n = cfg["ngram"]
    idx, size = collections.defaultdict(set), {}
    for name in cfg["benchmarks"]:
        for t in load(name):
            for part, text in _parts(t):
                toks = _norm(str(text)) if text else []
                if len(toks) < n:  # too short to match an n-gram, "True" would hit everywhere
                    continue
                key = (t.benchmark, t.task_id, t.mode, part)
                size[key] = len(s := _shingles(toks, n))
                for h in s:
                    idx[h].add(key)
    return idx, size


def _match(text: str, idx, size, cfg: dict) -> dict | None:
    cnt = collections.Counter(k for h in _shingles(_norm(text), cfg["ngram"]) for k in idx.get(h, ()))
    key, score = max(((k, c / size[k]) for k, c in cnt.items()), key=lambda kv: kv[1], default=(None, 0.0))
    if score < cfg["threshold"]:
        return None
    return {"check": "ngram", "benchmark": key[0], "task_id": key[1], "mode": key[2], "part": key[3],
            "score": round(score, 3)}


def decontaminate(rows: list[ManifestRow], contents: dict[str, str], cfg: dict):
    """Returns (rows with decontam_hit / kept / filter_reason set, report entries)."""
    idx, size = _index(cfg)
    report, out = [], []
    for r in rows:
        m = _match(contents[r.id], idx, size, cfg) if r.kept else None  # already dropped by an earlier step
        if m:
            report.append({"id": r.id, "path": r.path, "source": r.source, **m, "action": "dropped"})
            r = replace(r, decontam_hit=True, kept=False, filter_reason="decontam")
        out.append(r)
    return out, report


def run(src: Path, report_path: Path, cfg: dict) -> int:
    """Checks every record of a jsonl (corpus.jsonl `content`, or sft.jsonl `prompt` + `completion`)."""
    idx, size = _index(cfg)
    hits, n = [], 0
    for n, line in enumerate(open(src), 1):
        rec = json.loads(line)
        text = "\n".join(str(rec[k]) for k in ("content", "prompt", "completion") if k in rec)
        if m := _match(text, idx, size, cfg):
            hits.append({"line": n, "id": rec.get("id") or rec.get("task_id"), **m})
    by_benchmark = dict(collections.Counter(h["benchmark"] for h in hits))
    report_path.write_text(json.dumps({"checked": n, "hits": hits, "by_benchmark": by_benchmark}, indent=1))
    print(f"{src}: checked {n}, hits {len(hits)} {by_benchmark}")
    return int(bool(hits))


ALIAS = "decontaminated"  # what configs/train/sft.yaml pulls


def check_artifact(art, cfg: dict, report_path: Path) -> int:
    """Checks the sft.jsonl of a sft-dataset version; adds the `decontaminated` alias to it only if there are no hits."""
    rc = run(Path(art.download()) / "sft.jsonl", report_path, cfg)
    if rc == 0:
        art.aliases.append(ALIAS)
        art.save()
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--report", type=Path, default=Path("decontam_report.json"))
    ap.add_argument("--artifact", help="W&B sft-dataset version to check, e.g. sft-dataset:latest")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path("configs/data/decontam.yaml").read_text())
    if not a.artifact:
        sys.exit(run(a.input, a.report, cfg))

    import wandb
    from onprem_codegen.utils import init_run
    with init_run("decontaminate", cfg, f"decontaminate-{a.artifact.replace(':', '-')}") as r:
        r.use_artifact(a.artifact)  # lineage
        api_art = wandb.Api().artifact(f"{r.entity}/{r.project}/{a.artifact}")
        rc = check_artifact(api_art, cfg, a.report)
        rep = wandb.Artifact("decontam-report", type="report", metadata={"dataset": api_art.name, "passed": rc == 0})
        rep.add_file(str(a.report))
        r.log_artifact(rep)
    sys.exit(rc)


if __name__ == "__main__":
    main()
