"""
onprem_codegen/eval/sandbox.py: launches sandbox_runner.py for one job and returns its result.

Backends (env EXECUTE_BACKEND):
    docker  (default) the eval-sandbox image: no network, read-only rootfs, dropped caps,
                      pids/memory/cpu limits. Mirrors the eval-sandbox service in docker-compose.yml.
                      Build the image first (`docker compose build` also needs VLLM_VERSION in .env):
                          docker build -f docker/eval-sandbox.Dockerfile -t onprem-codegen/eval-sandbox .
    local             runs the runner in a plain subprocess on this machine. NOT a sandbox: use it
                      only on a disposable host (a Vast.ai instance with cobc installed natively)
                      or for trusted code such as benchmark canonical solutions.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

RUNNER = Path(__file__).with_name("sandbox_runner.py")
DEFAULT_IMAGE = "onprem-codegen/eval-sandbox"
STARTUP_S = 30.0   # docker start-up allowance on top of the job's own timeouts


class SandboxError(RuntimeError):
    """The sandbox itself failed (docker missing, image not built, ...), not the generated code."""


def run_job(job: dict, budget_s: float, backend: str | None = None) -> dict:
    backend = backend or os.environ.get("EXECUTE_BACKEND", "docker")
    workdir = Path(tempfile.mkdtemp(prefix="onprem-exec-"))
    try:
        (workdir / "job.json").write_text(json.dumps(job), encoding="utf-8")
        os.chmod(workdir, 0o777)   # the container user (uid 10001) must be able to write here
        if backend == "docker":
            _run_docker(workdir, budget_s)
        elif backend == "local":
            _run_local(workdir, budget_s)
        else:
            raise ValueError(f"unknown EXECUTE_BACKEND: {backend}")
        result = workdir / "result.json"
        if not result.is_file():
            raise SandboxError(f"runner produced no result.json (backend={backend})")
        return json.loads(result.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_local(workdir: Path, budget_s: float) -> None:
    try:
        p = subprocess.run([sys.executable, str(RUNNER), str(workdir / "job.json")],
                           capture_output=True, text=True, timeout=budget_s)
    except subprocess.TimeoutExpired as e:
        raise SandboxError(f"local runner exceeded its {budget_s:.0f}s budget") from e
    if p.returncode != 0:
        raise SandboxError(f"local runner failed:\n{p.stderr[-2000:]}")


def _docker_args(workdir: Path, name: str, image: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none", "--read-only", "--tmpfs", "/tmp:size=256m,exec",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--pids-limit", "128", "--memory", "1g", "--cpus", "1",
        "-v", f"{workdir}:/work:rw", "-v", f"{RUNNER}:/opt/sandbox_runner.py:ro",
        "-w", "/work", image,
        "python", "/opt/sandbox_runner.py", "/work/job.json",
    ]


def _run_docker(workdir: Path, budget_s: float) -> None:
    name = f"onprem-exec-{uuid.uuid4().hex[:12]}"
    image = os.environ.get("EVAL_SANDBOX_IMAGE", DEFAULT_IMAGE)
    try:
        p = subprocess.run(_docker_args(workdir, name, image), capture_output=True, text=True,
                           timeout=budget_s + STARTUP_S)
    except FileNotFoundError as e:
        raise SandboxError("docker is not installed; use EXECUTE_BACKEND=local on a disposable host") from e
    except subprocess.TimeoutExpired as e:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise SandboxError(f"sandbox container exceeded its {budget_s:.0f}s budget") from e
    if p.returncode != 0:
        raise SandboxError(f"docker run failed (is the image built? `docker build -f docker/eval-sandbox.Dockerfile "
                           f"-t {image} .`):\n{p.stderr[-2000:]}")
