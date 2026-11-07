#!/usr/bin/env bash
# Serve `base` + the LoRA adapter `cobol` on a fresh Vast instance (1x80 GB, PyTorch template). Run it in tmux.
# vLLM goes into its own venv: it needs torch==2.13.0, which differs from the training image.
# Needs env: WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, HF_TOKEN; run from the repo root.
#   ADAPTER=sft-v1-adapter:candidate bash scripts/vast_serve.sh      # default: sft-v0-adapter:candidate
# On the laptop:  ssh -p <PORT> root@<HOST> -N -L 8000:localhost:8000  then  python -m onprem_codegen.serve.check
set -euo pipefail

VENV=/workspace/vllm-env
ADAPTER=${ADAPTER:-sft-v0-adapter:candidate}
export HF_HOME=${HF_HOME:-/workspace/hf-cache}

pip install -q uv
[ -d "$VENV" ] || uv venv "$VENV"
uv pip install -q --python "$VENV/bin/python" "vllm==${VLLM_VERSION:-0.32.0}" torch==2.13.0 wandb
"$VENV/bin/vllm" --version   # logged: the vLLM version goes into the project log

rm -rf /adapters/cobol
"$VENV/bin/wandb" artifact get "$WANDB_ENTITY/$WANDB_PROJECT/$ADAPTER" --root /adapters/cobol
# Text-only adapter keys lack `language_model`: without this vLLM loads it but applies nothing (cobol == base).
PYTHONPATH=src python -m onprem_codegen.serve.adapter_keys /adapters/cobol

PATH="$VENV/bin:$PATH" exec bash docker/serve-entrypoint.sh   # MAX_NUM_SEQS=256 is its default
