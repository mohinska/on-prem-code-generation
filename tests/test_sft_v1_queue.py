"""scripts/sft_v1_queue.sh: DRY_RUN=1 prints every step's commands in order, runs nothing (no GPU, network, W&B)."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "sft_v1_queue.sh"
SECRETS = {"WANDB_API_KEY": "dummy-wandb-secret", "HF_TOKEN": "dummy-hf-secret", "CONTAINER_API_KEY": "dummy-vast-secret"}


def _run(tmp_path, **env):
    full = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "WANDB_ENTITY": "ent", "WANDB_PROJECT": "proj",
            "DRY_RUN": "1", "QUEUE_DIR": str(tmp_path / "q"), **SECRETS, **env}
    return subprocess.run(["bash", str(SCRIPT)], env=full, capture_output=True, text=True, cwd=tmp_path)


def _steps(out):
    return [line.split()[-1] for line in out.splitlines() if line.startswith("progress step ")]


def _cmds(out):
    return [line[2:] for line in out.splitlines() if line.startswith("+ ")]


def _no_secrets(r):
    for value in SECRETS.values():
        assert value not in r.stdout + r.stderr


def test_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_base_role_serves_base_only_and_generates_baseline_v2_and_heldout_base(tmp_path):
    r = _run(tmp_path, ROLE="base", AUTODESTROY="1", CONTAINER_ID="123")
    assert r.returncode == 0, r.stderr
    assert _steps(r.stdout) == ["benchmarks", "task-pool", "vllm-env", "serve", "check",
                                "generate-baseline-v2", "heldout-base", "verify-wandb", "destroy"]
    cmds = _cmds(r.stdout)
    assert not any("onprem_codegen.train.sft" in c for c in cmds)
    serve = next(c for c in cmds if "serve-entrypoint.sh" in c)
    assert "ADAPTER_PATH=/nonexistent" in serve and "EXTRA_LORA_MODULES" not in serve
    assert "check_served base" in cmds
    assert any("eval.run --config configs/eval/baseline-v2.yaml --stage generate" in c for c in cmds)
    assert any("eval.heldout generate --models base" in c for c in cmds)
    assert any("--alias heldout-base" in c for c in cmds)
    verify = next(c for c in cmds if "verify" in c)
    assert "eval-generations:baseline-v2" in verify and "eval-generations:heldout-base" in verify
    _no_secrets(r)


def test_adapter_role_trains_two_seeds_then_serves_both_adapters_and_generates(tmp_path):
    r = _run(tmp_path, ROLE="adapter")
    assert r.returncode == 0, r.stderr
    assert _steps(r.stdout) == ["train-sft-v1", "train-sft-v1-seed2", "benchmarks", "task-pool", "vllm-env",
                                "adapters", "serve", "check", "generate-sft-v1", "heldout-sft-v1",
                                "generate-sft-v1-seed2", "heldout-sft-v1-seed2"]   # no AUTODESTROY: no verify/destroy
    cmds = _cmds(r.stdout)
    train = [c for c in cmds if "onprem_codegen.train.sft" in c]
    assert len(train) == 2
    for c, name, seed in zip(train, ("sft-v1", "sft-v1-seed2"), (42, 43)):
        assert f"run_name={name} " in c and f"seed={seed} " in c
        assert "data.split_seed=42" in c          # same eval split for both seeds: only training noise differs
        assert "--config-name sft" in c and "data.artifact=sft-dataset:decontaminated-v1" in c
    assert any("sft-v1-adapter:candidate --root /adapters/cobol" in c for c in cmds)
    assert any("sft-v1-seed2-adapter:candidate --root /adapters/cobol-seed2" in c for c in cmds)
    assert sum("serve.adapter_keys" in c for c in cmds) == 2
    serve = next(c for c in cmds if "serve-entrypoint.sh" in c)
    assert "ADAPTER_PATH=/adapters/cobol " in serve and "EXTRA_LORA_MODULES=cobol-seed2=/adapters/cobol-seed2" in serve
    assert "check_served base cobol cobol-seed2" in cmds   # serve.check: all answer, both adapters change logprobs
    order = [c for c in cmds if "eval.run" in c or "eval.heldout" in c]
    assert "configs/eval/sft-v1.yaml" in order[0] and "--models cobol" in order[1] and "cobol-seed2" not in order[1]
    assert "configs/eval/sft-v1-seed2.yaml" in order[2] and "--models cobol-seed2" in order[3]
    _no_secrets(r)


def test_adapter_role_autodestroy_verifies_all_artifacts_first(tmp_path):
    r = _run(tmp_path, ROLE="adapter", AUTODESTROY="1", CONTAINER_ID="123")
    assert r.returncode == 0, r.stderr
    assert _steps(r.stdout)[-2:] == ["verify-wandb", "destroy"]
    verify = next(c for c in _cmds(r.stdout) if "verify" in c)
    for ref in ("sft-v1-adapter:candidate", "sft-v1-seed2-adapter:candidate", "eval-generations:sft-v1",
                "eval-generations:sft-v1-seed2", "eval-generations:heldout-sft-v1", "eval-generations:heldout-sft-v1-seed2"):
        assert ref in verify
    _no_secrets(r)


def test_done_markers_skip_steps(tmp_path):
    (tmp_path / "q" / "base" / "done").mkdir(parents=True)
    (tmp_path / "q" / "base" / "done" / "benchmarks").touch()
    r = _run(tmp_path, ROLE="base")
    assert "benchmarks" not in _steps(r.stdout) and "skip benchmarks (done)" in r.stdout


@pytest.mark.parametrize("missing", ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "HF_TOKEN", "ROLE"])
def test_missing_env_stops_before_any_step(tmp_path, missing):
    r = _run(tmp_path, **{"ROLE": "base", missing: ""})
    assert r.returncode != 0 and missing in r.stdout + r.stderr and not _steps(r.stdout)
    _no_secrets(r)


def test_destroy_skips_the_confirmation_prompt_and_fails_loudly_if_the_instance_survives(tmp_path):
    # `vastai destroy instance` asks "[y/N]"; without -y it hangs in tmux or prints "Aborted." and exits 0 (sft-v0 run).
    r = _run(tmp_path, ROLE="base", AUTODESTROY="1", CONTAINER_ID="123")
    destroy = r.stdout[r.stdout.index("progress step 9/9 destroy"):]
    assert "destroy instance $CONTAINER_ID -y" in destroy and "< /dev/null" in destroy
    assert "/workspace/vllm-env/bin/vastai" in destroy          # own venv: system pip may be externally managed
    assert "still running" in destroy                           # the destroy is checked, not assumed
