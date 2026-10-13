# Pipeline entry points (C). Run from the repo root.
PY ?= python
export PYTHONPATH := src

.PHONY: test codenet data sft-data sft-data-dry train train-smoke train-dry

test:
	$(PY) -m pytest -q

# CodeNet parts -> data/raw/codenet_cobol/, task-pool artifact
codenet:
	$(PY) -m onprem_codegen.data.download_codenet

# raw-corpus (Stack v1 + X-COBOL + CodeNet) -> clean-corpus
data: codenet
	$(PY) -m onprem_codegen.data.collect --codenet-root data/raw/codenet_cobol/Project_CodeNet
	$(PY) -m onprem_codegen.data.clean

# Rejection sampling on A's vLLM server (SSH tunnel on :8000) -> sft-dataset
sft-data:
	$(PY) -m onprem_codegen.data.rejection_sampling $(ARGS)

# Same pipeline with canned generations and the stub executor; nothing goes to W&B
sft-data-dry:
	EXECUTE_MODE=stub $(PY) -m onprem_codegen.data.rejection_sampling --dry-run --no-wandb \
		--task-pool tests/fixtures/task_pool_tiny.jsonl --out data/raw/sft-dataset-dry

train-smoke:
	$(PY) -m onprem_codegen.train.sft

# make train ARGS="run_name=sft-v1"
train:
	$(PY) -m onprem_codegen.train.sft --config-name sft $(ARGS)

train-dry:
	$(PY) -m onprem_codegen.train.sft model.tiny=true run_name=sft-smoke-local train.max_steps=6 train.save_steps=3
