"""Flag corpus files that overlap COBOLEval tasks (prompt + tests), by n-gram containment."""
import collections
import json
import re
from dataclasses import replace
from pathlib import Path

from onprem_codegen.data.schema import ManifestRow


def _norm(text: str) -> list[str]:
    lines = [re.sub(r"\*>.*", "", l) for l in text.splitlines() if not (len(l) > 6 and l[6] in "*/")]
    return " ".join(lines).upper().split()


def _shingles(toks: list[str], n: int) -> set[int]:
    return {hash(" ".join(toks[i:i + n])) for i in range(max(len(toks) - n + 1, 1))}


def decontaminate(rows: list[ManifestRow], contents: dict[str, str], cfg: dict):
    """Returns (rows with decontam_hit / kept / filter_reason set, report entries)."""
    n = cfg["ngram"]
    bench = [json.loads(l) for l in open(cfg["benchmark"])]
    idx, size = collections.defaultdict(set), {}
    for b in bench:
        s = _shingles(_norm(b["prompt"]) + sum((_norm(t["test"]) for t in b["tests"]), []), n)
        size[b["task_id"]] = len(s)
        for h in s:
            idx[h].add(b["task_id"])

    report, out = [], []
    for r in rows:
        if not r.kept:  # already dropped by an earlier step
            out.append(r)
            continue
        text = contents[r.id]
        cnt = collections.Counter(t for h in _shingles(_norm(text), n) for t in idx.get(h, ()))
        task, hits = max(cnt.items(), key=lambda kv: kv[1], default=(None, 0))
        score = hits / size[task] if task else 0.0
        if score >= cfg["threshold"]:
            report.append({"id": r.id, "path": r.path, "source": r.source, "check": "ngram",
                           "task_id": task, "score": round(score, 3), "action": "dropped"})
            out.append(replace(r, decontam_hit=True, kept=False, filter_reason="decontam"))
            continue
        out.append(r)
    return out, report
