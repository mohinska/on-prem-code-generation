#!/usr/bin/env bash
# Unattended queue on the GPU instance: train sft-v0 -> serve it -> generate the non-thinking eval (plan 2.1, 3.1).
# Stops at the first failure, writes a marker per finished step in .overnight/ (re-running skips them), and leaves the
# vLLM server up at the end. Execution of the generations (sandbox, Docker) runs on the laptop afterwards.
# Does NOT destroy the instance: do that in the Vast console when it is done.
#
#   tmux new -s night                      # or attach to the existing one
#   cd /workspace/on-prem-code-generation && bash scripts/overnight.sh 2>&1 | tee overnight.log
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf-cache} PYTHONPATH=$PWD/src
mkdir -p .overnight
say() { echo "=== $(date -u +%H:%M:%S) UTC  $*"; }
trap 'say "FAILED at line $LINENO (see sft-v0.log / serve.log / overnight.log); queue stopped"' ERR
step() { [ -f ".overnight/$1" ] && { say "skip $1 (done)"; return 1; } || return 0; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 2000 ] && ! curl -sf localhost:8000/v1/models > /dev/null; then
  echo "GPU has ${used} MiB in use by something that is not our server: stop it first"; exit 1
fi

if step train; then
  say "1/3 train sft-v0 (3 epochs, ~40 min)"
  if curl -sf localhost:8000/v1/models > /dev/null; then echo "vLLM is still running: stop it before training"; exit 1; fi
  python -m onprem_codegen.train.sft --config-name sft run_name=sft-v0 \
    train.num_train_epochs=3 train.save_steps=44 train.logging_steps=5 > sft-v0.log 2>&1
  touch .overnight/train
fi

if step benchmarks; then
  say "2/3 fetch pinned benchmarks"
  python -m onprem_codegen.eval.benchmarks --wandb
  touch .overnight/benchmarks
fi

say "3/3 serve sft-v0-adapter as 'cobol'"
if ! curl -sf localhost:8000/v1/models > /dev/null; then
  ADAPTER=sft-v0-adapter:candidate nohup bash scripts/vast_serve.sh > serve.log 2>&1 &
  for _ in $(seq 1 240); do                     # up to 40 min for weights + CUDA graphs
    curl -sf localhost:8000/v1/models > /dev/null && break
    sleep 10
  done
fi
curl -sf localhost:8000/v1/models > /dev/null || { echo "server did not come up, see serve.log"; exit 1; }
python -m onprem_codegen.serve.check          # both answer AND the adapter changes the logprobs, else stop here

if step generate; then
  say "generate non-thinking eval (3-5 h, resumable)"
  python -m onprem_codegen.eval.run --config configs/eval/sft-v0.yaml --stage generate
  touch .overnight/generate
fi
say "queue done: generations are in W&B (eval-generations, run sft-v0-generate); execute them on the laptop"
