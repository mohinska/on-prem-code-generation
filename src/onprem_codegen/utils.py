import json
import subprocess
from pathlib import Path


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    """[] if the file is missing; a half-written last line (crash mid-write) is skipped, so that record re-runs."""
    rows = []
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows
