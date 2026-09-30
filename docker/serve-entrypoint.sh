#!/usr/bin/env bash
# Serves the base model and, if present, the LoRA adapter side by side,
# so the demo can toggle base <-> adapted by switching the `model` field.
set -euo pipefail

args=(
  "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ -d "$ADAPTER_PATH" ]]; then
  args+=(--enable-lora --max-lora-rank "$MAX_LORA_RANK"
         --lora-modules "$ADAPTER_NAME=$ADAPTER_PATH")
else
  echo "No adapter at $ADAPTER_PATH; serving base model only." >&2
fi

exec vllm serve "${args[@]}" "$@"
