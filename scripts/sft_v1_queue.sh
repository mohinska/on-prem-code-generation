#!/usr/bin/env bash
# One-off sft-v1 queue on a fresh Vast instance (plan 2.1-2.3). Two roles, so two instances can run in parallel:
#   ROLE=base     serve `base` only -> baseline v2 generations + held-out generations for base (no training)
#   ROLE=adapter  train sft-v1 (seed 42) and sft-v1-seed2 (seed 43) -> serve base + cobol + cobol-seed2 ->
#                 sft-v1 generations (3 benchmarks) + held-out for cobol; COBOLEval sampled + held-out for cobol-seed2
# Stops at the first failure; a marker per finished step in $QUEUE_DIR/<role>/done/ (a re-run skips those steps).
# Generations end in W&B (`eval-generations`, aliases = run names); execute them locally in the sandbox afterwards.
# AUTODESTROY=1 destroys the instance after W&B holds every artifact of the role (needs CONTAINER_ID,
# CONTAINER_API_KEY, which Vast sets). DRY_RUN=1 prints every step's commands in order and runs nothing.
# Needs env WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, HF_TOKEN; vLLM goes into its own venv (as vast_serve.sh).
#
#   tmux new -s q
#   cd /workspace/on-prem-code-generation && ROLE=adapter AUTODESTROY=1 bash scripts/sft_v1_queue.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ROLE=${ROLE:-}
for v in ROLE WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT HF_TOKEN; do
  [ -n "${!v:-}" ] || { echo "missing env var $v (ROLE=base|adapter; W&B and HF keys from the Vast account)"; exit 1; }
done
[[ $ROLE == base || $ROLE == adapter ]] || { echo "ROLE must be base or adapter, got '$ROLE'"; exit 1; }
DRY=${DRY_RUN:-0}
AUTO=${AUTODESTROY:-0}
if [ "$AUTO" = 1 ]; then
  for v in CONTAINER_ID CONTAINER_API_KEY; do [ -n "${!v:-}" ] || { echo "AUTODESTROY=1 needs $v"; exit 1; }; done
fi

export HF_HOME=${HF_HOME:-/workspace/hf-cache} PYTHONPATH=$PWD/src
Q=${QUEUE_DIR:-$PWD/outputs/sft-v1-queue}/$ROLE
VENV=/workspace/vllm-env
URL=http://localhost:8000/v1
P=$WANDB_ENTITY/$WANDB_PROJECT
DATA=sft-dataset:decontaminated-v1
# ~662 rows, 10% of tasks held out for eval -> ~596 train rows / (1 x 16) = ~37 steps per epoch: checkpoint + eval
# loss once per epoch (3 epochs ~ 112 steps).
SAVE_STEPS=37

say() { echo "progress $(date -u +%H:%M:%S) $*"; }
x() { if [ "$DRY" = 1 ]; then echo "+ $*"; else "$@"; fi; }   # every command of a step goes through x
up() { curl -sf "$URL/models" > /dev/null; }

# ---------- steps (one function each; output goes to $Q/logs/<step>.log) ----------

train() {   # run_name seed
  if [ "$DRY" != 1 ] && up; then echo "vLLM is running on :8000: stop it before training"; return 1; fi
  x python -m onprem_codegen.train.sft --config-name sft run_name="$1" seed="$2" data.split_seed=42 data.artifact=$DATA \
    train.save_steps=$SAVE_STEPS
}
step_train_sft_v1() { train sft-v1 42; }
step_train_sft_v1_seed2() { train sft-v1-seed2 43; }

step_benchmarks() { x python -m onprem_codegen.eval.benchmarks --wandb; }
step_task_pool() { x wandb artifact get "$P/task-pool:v1" --root artifacts/task-pool:v1; }   # what eval.heldout reads

step_vllm_env() {
  x pip install -q uv
  [ -d "$VENV" ] || x uv venv "$VENV"
  x uv pip install -q --python "$VENV/bin/python" "vllm==${VLLM_VERSION:-0.32.0}" torch==2.13.0
  x "$VENV/bin/vllm" --version   # goes into the project log
}

step_adapters() {
  for pair in cobol=sft-v1 cobol-seed2=sft-v1-seed2; do
    x rm -rf "/adapters/${pair%%=*}"
    x wandb artifact get "$P/${pair#*=}-adapter:candidate" --root "/adapters/${pair%%=*}"
    # Text-only adapter keys lack `language_model`: without this vLLM loads it but applies nothing (cobol == base).
    x python -m onprem_codegen.serve.adapter_keys "/adapters/${pair%%=*}"
  done
}

if [ "$ROLE" = base ]; then
  SERVE=(env ADAPTER_PATH=/nonexistent)   # no adapter dir -> serve-entrypoint.sh serves base only
else
  SERVE=(env ADAPTER_PATH=/adapters/cobol EXTRA_LORA_MODULES=cobol-seed2=/adapters/cobol-seed2)
fi
SERVE+=(PATH="$VENV/bin:$PATH" bash docker/serve-entrypoint.sh)
[ "$ROLE" = adapter ] && SERVE+=(--max-loras 2)
step_serve() {
  if [ "$DRY" = 1 ]; then echo "+ nohup ${SERVE[*]//"$PATH"/\$PATH} > $Q/logs/vllm.log 2>&1 &"; return; fi
  if up; then echo "vLLM already up on :8000"; return; fi
  nohup "${SERVE[@]}" > "$Q/logs/vllm.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 240); do   # up to 40 min for weights + CUDA graphs
    up && return 0
    kill -0 $pid 2> /dev/null || { echo "vLLM exited, see $Q/logs/vllm.log"; return 1; }
    [ $((i % 6)) = 0 ] && say "waiting for vLLM ($((i / 6)) min)"
    sleep 10
  done
  echo "vLLM did not come up in 40 min, see $Q/logs/vllm.log"; return 1
}

check_served() {   # serve.check: every model answers; every adapter (args after base) changes the logprobs
  python - "$@" <<'PY'
import sys
from onprem_codegen.serve.check import adapter_effect, check
url, models = "http://localhost:8000/v1", sys.argv[1:]
res = check(url, models)
for m, err in res.items():
    print(f"{m:12} {'ok' if err is None else 'FAIL ' + err}")
assert not any(res.values()), "a served model failed"
for m in models[1:]:
    effect = adapter_effect(url, m)
    print(f"{m:12} adapter effect (max logprob diff) {effect:.4f}")
    assert effect > 1e-3, f"{m}: adapter is not applied"
PY
}
if [ "$ROLE" = base ]; then step_check() { x check_served base; }
else step_check() { x check_served base cobol cobol-seed2; }; fi

gen() {   # resumable generate: re-run until no request failed (each run retries only the missing ones)
  if [ "$DRY" = 1 ]; then echo "+ $*"; return; fi
  for _ in 1 2 3; do
    "$@" | tee "$Q/logs/last-gen.out"
    grep -q "requests failed" "$Q/logs/last-gen.out" || return 0
  done
  echo "requests still failing after 3 runs: $*"; return 1
}
evalgen() { gen python -m onprem_codegen.eval.run --config "configs/eval/$1.yaml" --stage generate; }
heldout() {   # served model, W&B alias
  gen python -m onprem_codegen.eval.heldout generate --models "$1"
  x wandb artifact put "outputs/eval/heldout/$1/generations.jsonl" --name "$P/eval-generations" \
    --type generations --alias "$2"
}
step_generate_baseline_v2() { evalgen baseline-v2; }
step_heldout_base() { heldout base heldout-base; }
step_generate_sft_v1() { evalgen sft-v1; }
step_heldout_sft_v1() { heldout cobol heldout-sft-v1; }
step_generate_sft_v1_seed2() { evalgen sft-v1-seed2; }
step_heldout_sft_v1_seed2() { heldout cobol-seed2 heldout-sft-v1-seed2; }

verify_wandb() {
  python - "$@" <<'PY'
import os, sys, wandb
api, p = wandb.Api(), f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
for ref in sys.argv[1:]:
    assert api.artifact(f"{p}/{ref}").state == "COMMITTED", ref
    print("in W&B:", ref)
PY
}
if [ "$ROLE" = base ]; then
  EXPECTED=(eval-generations:baseline-v2 eval-generations:heldout-base)
else
  EXPECTED=(sft-v1-adapter:candidate sft-v1-seed2-adapter:candidate eval-generations:sft-v1
            eval-generations:sft-v1-seed2 eval-generations:heldout-sft-v1 eval-generations:heldout-sft-v1-seed2)
fi
step_verify_wandb() { x verify_wandb "${EXPECTED[@]}"; }
# `vastai destroy instance` asks "[y/N]": without -y it waits on stdin in tmux, or reads EOF, prints "Aborted." and
# exits 0, so nothing is destroyed and nothing fails (that is what happened after sft-v0). If this process is still
# alive 3 min after the call, the destroy did not happen: fail loudly so the instance is not left running unnoticed.
step_destroy() {
  local cmd='$VENV/bin/vastai destroy instance $CONTAINER_ID -y --api-key $CONTAINER_API_KEY < /dev/null'
  if [ "$DRY" = 1 ]; then
    echo "+ uv pip install -q --python $VENV/bin/python vastai"
    echo "+ ${cmd//\$VENV/$VENV}"
    echo "+ sleep 180; echo 'instance still running: destroy it by hand'; exit 1"
    return
  fi
  uv pip install -q --python "$VENV/bin/python" vastai
  "$VENV/bin/vastai" destroy instance "$CONTAINER_ID" -y --api-key "$CONTAINER_API_KEY" < /dev/null
  sleep 180
  echo "instance $CONTAINER_ID still running 3 min after destroy: destroy it by hand (console or vastai)"; return 1
}

# ---------- queue ----------

if [ "$ROLE" = base ]; then
  STEPS=(benchmarks task-pool vllm-env serve check generate-baseline-v2 heldout-base)
else
  STEPS=(train-sft-v1 train-sft-v1-seed2 benchmarks task-pool vllm-env adapters serve check
         generate-sft-v1 heldout-sft-v1 generate-sft-v1-seed2 heldout-sft-v1-seed2)
fi
[ "$AUTO" = 1 ] && STEPS+=(verify-wandb destroy)
ALWAYS=" serve check verify-wandb destroy "   # no marker: the server must be up again after a restart

if [ "$DRY" != 1 ]; then
  mkdir -p "$Q/done" "$Q/logs"
  exec > >(tee -a "$Q/queue.log") 2>&1
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -gt 2000 ] && ! up; then echo "GPU has ${used} MiB in use by something that is not our server"; exit 1; fi
fi
echo "sft-v1 queue, ROLE=$ROLE, steps: ${STEPS[*]}"
echo "watch:  tail -f $Q/queue.log"
echo "        grep -h progress $Q/queue.log $Q/logs/*.log | tail"
echo "        tail -f $Q/logs/<step>.log          # current step's output (training, N/M requests)"

CUR=
trap 'rc=$?; [ $rc = 0 ] || echo "FAILED at step ${CUR:-setup} (exit $rc), see $Q/logs/${CUR:-}.log; queue stopped"' EXIT
n=${#STEPS[@]}
for i in "${!STEPS[@]}"; do
  CUR=${STEPS[$i]}
  if [ -f "$Q/done/$CUR" ]; then say "skip $CUR (done)"; continue; fi
  echo "progress step $((i + 1))/$n $CUR"
  if [ "$DRY" = 1 ]; then
    "step_${CUR//-/_}"
  else
    say "start $CUR" >> "$Q/logs/$CUR.log"
    "step_${CUR//-/_}" >> "$Q/logs/$CUR.log" 2>&1
    [[ $ALWAYS == *" $CUR "* ]] || touch "$Q/done/$CUR"
    say "done $CUR"
  fi
done
CUR=
say "queue done (ROLE=$ROLE); execute locally in the sandbox:"
echo "  python -m onprem_codegen.eval.run --config configs/eval/<run>.yaml --stage execute --generations eval-generations:<run>"
echo "  wandb artifact get $P/eval-generations:heldout-<model> --root outputs/eval/heldout/<served model>  # then eval.heldout execute"
