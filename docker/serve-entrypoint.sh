#!/usr/bin/env bash
# Serves the base model and, if present, the LoRA adapter side by side,
# so the demo can toggle base <-> adapted by switching the `model` field.
# Defaults match serve.Dockerfile, so it also runs straight on a vllm/vllm-openai image (Vast).
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3.8-27B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-base}
ADAPTER_NAME=${ADAPTER_NAME:-cobol}
ADAPTER_PATH=${ADAPTER_PATH:-/adapters/cobol}
MAX_LORA_RANK=${MAX_LORA_RANK:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
# vLLM default 1024 exceeds the Mamba cache blocks of this model on 80 GB and fails at startup.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
PORT=${PORT:-8000}

args=(
  "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
)

if [[ -d "$ADAPTER_PATH" ]]; then
  args+=(--enable-lora --max-lora-rank "$MAX_LORA_RANK"
         --lora-modules "$ADAPTER_NAME=$ADAPTER_PATH")
else
  echo "No adapter at $ADAPTER_PATH; serving base model only." >&2
fi

exec vllm serve "${args[@]}" "$@"
