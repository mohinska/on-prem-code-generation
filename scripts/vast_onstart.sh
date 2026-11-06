#!/usr/bin/env bash
# Bootstrap a fresh Vast.ai training instance (image: pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel).
# Idempotent: safe to re-run after a restart. Log: /var/log/onstart.log when passed via --onstart.
#
# Needs Vast account env vars: GITHUB_TOKEN (read-only, this repo), WANDB_API_KEY, WANDB_ENTITY,
# WANDB_PROJECT, HF_TOKEN. Optional: BRANCH (default main).
set -euo pipefail

REPO_DIR=/workspace/on-prem-code-generation
BRANCH=${BRANCH:-main}
export HF_HOME=/workspace/hf-cache
export DEBIAN_FRONTEND=noninteractive
# The image's torch lives in the system Python (PEP 668 "externally managed"); the container is disposable.
export PIP_BREAK_SYSTEM_PACKAGES=1

apt-get update -qq && apt-get install -y -qq git tmux ninja-build > /dev/null

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --branch "$BRANCH" \
        "https://x-access-token:${GITHUB_TOKEN}@github.com/mohinska/on-prem-code-generation.git" "$REPO_DIR"
    # Don't leave the token in .git/config.
    git -C "$REPO_DIR" remote set-url origin https://github.com/mohinska/on-prem-code-generation.git
fi

# causal-conv1d and flash-linear-attention build against the installed torch.
pip install -q --no-build-isolation -r "$REPO_DIR/requirements.txt"

# Env for interactive SSH sessions.
grep -q onprem-codegen ~/.bashrc || cat >> ~/.bashrc <<EOF
# onprem-codegen
export HF_HOME=$HF_HOME
export PYTHONPATH=$REPO_DIR/src
cd $REPO_DIR
EOF

# ~54 GB; resumes if interrupted. Vision / MTP weights come along but are never loaded.
hf download Qwen/Qwen3.8-27B > /dev/null

echo "onstart: done"
