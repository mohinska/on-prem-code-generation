"""X-COBOL (Zenodo v2) -> raw-corpus records + manifest rows (source=xcobol).

`root` = extracted dir with COBOL_Files/<owner>@<repo>/<file> and final_repo_statistics.csv.
Licenses come from the GitHub API (SPDX id of each repo), cached in <root>/licenses.json.
"""
import csv
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from onprem_codegen.data.schema import ManifestRow, raw_record


def _repo_license(repo_url: str, token: str | None) -> str | None:
    req = urllib.request.Request(
        repo_url.replace("https://github.com/", "https://api.github.com/repos/") + "/license",
        headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req) as r:
            spdx = json.load(r)["license"]["spdx_id"]
    except urllib.error.HTTPError:  # 404 = repo has no license file / repo gone
        return None
    return None if spdx == "NOASSERTION" else spdx


def load_xcobol(cfg, root: str | Path, out_dir: str | Path) -> list[ManifestRow]:
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = root / "licenses.json"
    lic = json.loads(cache.read_text()) if cache.exists() else {}
    with open(root / "final_repo_statistics.csv", newline="") as f:
        urls = {r["Repo Name"]: r["Repo URL"] for r in csv.DictReader(f)}
    token = os.environ.get("GITHUB_TOKEN")
    rows = []
    with open(out / "corpus.jsonl", "w") as f:
        for p in sorted((root / "COBOL_Files").rglob("*")):
            if not p.is_file():
                continue
            repo = p.relative_to(root / "COBOL_Files").parts[0]
            if repo not in lic:
                lic[repo] = _repo_license(urls[repo], token) if repo in urls else None
            text = p.read_bytes().decode("utf-8", errors="replace")
            rel = str(p.relative_to(root / "COBOL_Files"))
            rec, h, n = raw_record("xcobol", text, lic[repo], {"repo_name": repo, "original_path": rel, "repo_license": lic[repo]})
            f.write(json.dumps(rec) + "\n")
            rows.append(ManifestRow(
                id=h, source="xcobol", source_version=cfg["version"], path=rel, ext=p.suffix.lower(),
                repo=repo, sha256=h, license=lic[repo], license_ok=lic[repo] is not None, n_bytes=n))
    cache.write_text(json.dumps(lic))
    return rows
