# Pipeline entry points (C). Run from the repo root.
PY ?= python
export PYTHONPATH := src

.PHONY: test codenet data sft-data decontam train train-smoke train-dryrun benchmarks eval eval-generate eval-execute

test:
	$(PY) -m pytest -q

# CodeNet parts -> data/raw/codenet_cobol/, task-pool artifact
codenet:
	$(PY) -m onprem_codegen.data.download_codenet

# raw-corpus (Stack v1 + X-COBOL + CodeNet) -> clean-corpus
data: codenet
	$(PY) -m onprem_codegen.data.collect
	$(PY) -m onprem_codegen.data.clean

# Rejection sampling on A's vLLM server (SSH tunnel on :8000) -> sft-dataset
# Runs generated code in the eval-sandbox image: docker compose build eval-sandbox
sft-data:
	$(PY) -m onprem_codegen.data.rejection_sampling $(ARGS)

# Check a sft-dataset version against the benchmarks; no hits -> alias `decontaminated` (what `make train` pulls)
# make decontam ARGS="--artifact sft-dataset:v1"
decontam:
	$(PY) -m onprem_codegen.data.decontaminate --artifact sft-dataset:latest $(ARGS)

train-smoke:
	$(PY) -m onprem_codegen.train.sft

# B1 dry run: 27B at real length; kill after the step-10 checkpoint artifact, delete checkpoints/sft-dryrun, rerun
train-dryrun:
	$(PY) -m onprem_codegen.train.sft run_name=sft-dryrun train.max_steps=20 train.max_length=4096 train.save_steps=10 $(ARGS)

# make train ARGS="run_name=sft-v1"
train:
	$(PY) -m onprem_codegen.train.sft --config-name sft $(ARGS)

# Fetch + verify the pinned benchmarks, log them as benchmark-* W&B artifacts
benchmarks:
	$(PY) -m onprem_codegen.eval.benchmarks --wandb

# Eval: vLLM (SSH tunnel on :8000) -> eval-sandbox image -> W&B. configs/eval/baseline.yaml
# Build the sandbox once: docker build -f docker/eval-sandbox.Dockerfile -t onprem-codegen/eval-sandbox .
eval:
	$(PY) -m onprem_codegen.eval.run $(ARGS)

# Split when generation and execution run on different machines (Vast has no Docker)
eval-generate:
	$(PY) -m onprem_codegen.eval.run --stage generate $(ARGS)

# make eval-execute ARGS="--generations eval-generations:baseline"
eval-execute:
	$(PY) -m onprem_codegen.eval.run --stage execute $(ARGS)
