import json
import os
import subprocess
from pathlib import Path


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def init_run(job_type: str, config: dict, name: str | None = None):
    """wandb.init with the project from WANDB_PROJECT, plus git commit and Vast instance ID in the config."""
    import wandb
    commit = git_commit() or ""
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "onprem-codegen"), job_type=job_type, name=name or f"{job_type}-{commit[:7]}",
        config={**config, "git_commit": commit,
                "vast_instance": os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL")})


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
