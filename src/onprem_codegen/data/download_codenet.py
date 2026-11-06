"""IBM Project CodeNet -> data/raw/codenet_cobol/ (only the parts we use) -> task-pool artifact.

    python -m onprem_codegen.data.download_codenet [--archive PATH] [--force] [--no-wandb]

1. Streams Project_CodeNet.tar.gz (7.8 GB) from the URL in configs/data/codenet_cobol.yaml, or reads
   a local copy (--archive, e.g. after `wget -c`), and extracts only `extract` patterns. The full
   archive never touches disk. A marker file makes a re-run skip this step.
2. Builds data/raw/task-pool/{task_pool.jsonl, stats.json} (contracts.md: task-pool) and logs `task-pool`.

Accepted COBOL submissions reach raw-corpus through collect:
    python -m onprem_codegen.data.collect
"""
import argparse
import collections
import fnmatch
import hashlib
import json
import os
import re
import tarfile
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import yaml

from onprem_codegen.utils import git_commit

CFG = Path("configs/data/codenet_cobol.yaml")
RAW = Path("data/raw/codenet_cobol")
OUT = Path("data/raw/task-pool")
MARKER = ".extracted.json"


class _HashingReader:
    """File-like wrapper that hashes every byte read (the archive is never stored)."""

    def __init__(self, f, every=256 << 20, total=None):
        self.f, self.h, self.n, self.every, self.total, self.steps = f, hashlib.sha256(), 0, every, total, 0

    def read(self, size=-1):
        b = self.f.read(size)
        self.h.update(b)
        self.n += len(b)
        if self.n // self.every > self.steps:  # one line per `every` bytes, so a long download is visible
            self.steps = self.n // self.every
            print(f"downloaded {self.n / 1e9:.1f} GB / {self.total}", flush=True)
        return b


def extract(src, out_dir: Path, patterns: list[str], total: str | None = None) -> dict:
    """Stream a .tar.gz, write members matching `patterns` under out_dir. Returns sha256 + counts."""
    reader = _HashingReader(src, total=total)
    counts = collections.Counter()
    with tarfile.open(fileobj=reader, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile() or not any(fnmatch.fnmatch(m.name, p) for p in patterns):
                continue
            rel = Path(m.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe path in archive: {m.name}")
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(tf.extractfile(m).read())
            counts[rel.parts[1] if len(rel.parts) > 1 else rel.parts[0]] += 1
    while reader.read(1 << 20):  # hash the trailing padding too
        pass
    return {"sha256": reader.h.hexdigest(), "bytes": reader.n, "files_by_dir": dict(counts)}


def download(cfg: dict, archive: str | None, force: bool) -> dict:
    marker = RAW / MARKER
    if marker.exists() and not force:
        print(f"{marker} exists, skipping download (--force to redo)")
        return json.loads(marker.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    src = open(archive, "rb") if archive else urllib.request.urlopen(cfg["archive"]["url"])
    with src:
        info = extract(src, RAW, cfg["extract"], cfg["archive"]["size"])
    pinned = cfg["archive"]["sha256"]
    if pinned and info["sha256"] != pinned:
        raise RuntimeError(f"sha256 mismatch: got {info['sha256']}, configs pin {pinned}")
    info["date_downloaded"] = time.strftime("%Y-%m-%d")
    marker.write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return info


class _HtmlText(HTMLParser):
    """Problem HTML -> plain text. <pre> keeps its line breaks; AtCoder pages keep English only."""

    BLOCK = {"p", "div", "br", "pre", "li", "ul", "ol", "tr", "table", "section", "blockquote",
             "h1", "h2", "h3", "h4", "h5", "h6", "hr"}
    VOID = {"br", "img", "hr", "meta", "link", "input", "wbr", "col"}

    def __init__(self, skip_class: str | None):
        super().__init__(convert_charrefs=True)
        self.skip_class, self.parts, self.pre = skip_class, [], 0
        self.skip_tag, self.skip_depth = None, 0
        self.has_image = False

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.has_image = True
        if self.skip_tag:
            self.skip_depth += tag == self.skip_tag
            return
        classes = (dict(attrs).get("class") or "").split()
        if tag in ("script", "style") or (self.skip_class and self.skip_class in classes):
            if tag not in self.VOID:
                self.skip_tag, self.skip_depth = tag, 1
            return
        if tag == "pre":
            self.pre += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.skip_tag:
            if tag == self.skip_tag:
                self.skip_depth -= 1
                if self.skip_depth == 0:
                    self.skip_tag = None
            return
        if tag == "pre":
            self.pre = max(0, self.pre - 1)
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_tag:
            self.parts.append(data if self.pre else re.sub(r"\s+", " ", data))


def html_to_text(html: str) -> tuple[str, bool]:
    """Returns (plain text, has_image)."""
    p = _HtmlText(skip_class="lang-ja" if 'class="lang-en"' in html else None)
    p.feed(html)
    p.close()
    text = "\n".join(l.strip() for l in "".join(p.parts).split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, p.has_image


def _tests(io_dir: Path) -> list[dict]:
    """derived/input_output/data/<pid>/: input.txt + output.txt (also input_N.txt + output_N.txt)."""
    tests = []
    for inp in sorted(io_dir.glob("input*.txt")):
        out = io_dir / inp.name.replace("input", "output", 1)
        if out.exists():
            tests.append({"input": inp.read_text(errors="replace"), "output": out.read_text(errors="replace")})
    return tests


def build_task_pool(root: Path, cfg: dict) -> tuple[list[dict], dict]:
    """root = extracted Project_CodeNet/. Every problem with tests and a usable statement becomes a task."""
    io_root = root / "derived" / "input_output" / "data"
    reasons = collections.Counter()
    tasks = []
    pids = sorted(p.stem for p in (root / "metadata").glob("p[0-9]*.csv"))
    for pid in pids:
        tests = _tests(io_root / pid)
        html = root / "problem_descriptions" / f"{pid}.html"
        if len(tests) < cfg["min_tests"]:
            reasons["no_tests"] += 1
            continue
        if not html.exists():
            reasons["no_description"] += 1
            continue
        text, has_image = html_to_text(html.read_text(errors="replace"))
        non_ascii = sum(not c.isascii() for c in text) / max(len(text), 1)
        reason = ("has_image" if has_image and cfg["drop_with_images"] else
                  "too_short" if len(text) < cfg["min_chars"] else
                  "too_long" if len(text) > cfg["max_chars"] else
                  "non_ascii" if non_ascii > cfg["max_non_ascii_ratio"] else None)
        if reason:
            reasons[reason] += 1
            continue
        reasons["kept"] += 1
        tasks.append({"task_id": f"codenet/{pid}", "source": "codenet_cobol", "prompt": text,
                      "tests": tests, "meta": {"problem_id": pid}})

    stats = {"problems": len(pids), "problems_with_tests": sum(1 for p in pids if (io_root / p).is_dir()),
             "tasks": len(tasks), "by_reason": dict(reasons), "tests_per_task": dict(collections.Counter(
                 len(t["tests"]) for t in tasks))}
    return tasks, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", help="local Project_CodeNet.tar.gz instead of the URL")
    ap.add_argument("--force", action="store_true", help="re-extract even if the marker exists")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())

    info = download(cfg, a.archive, a.force)
    root = RAW / "Project_CodeNet"
    if not any((root / "derived" / "input_output" / "data").glob("*")):
        raise RuntimeError("no derived/input_output tests extracted; check `extract` paths against the archive")
    tasks, stats = build_task_pool(root, cfg["task_pool"])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "task_pool.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    stats["archive_sha256"] = info["sha256"]
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    if not a.no_wandb:
        import wandb
        with wandb.init(project=os.environ.get("WANDB_PROJECT", "onprem-codegen"), job_type="task-pool",
                        config={**cfg, "git_commit": git_commit()}) as run:
            run.summary.update({k: v for k, v in stats.items() if not isinstance(v, dict)})
            art = wandb.Artifact("task-pool", type="dataset", metadata={**stats, "codenet_version": cfg["version"]})
            art.add_dir(str(OUT))
            run.log_artifact(art)


if __name__ == "__main__":
    main()
