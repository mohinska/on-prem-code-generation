---
paths:
  - "src/onprem_codegen/data/**"
  - "src/onprem_codegen/eval/**"
  - "configs/data/**"
  - "configs/eval/**"
  - "data/**"
  - "benchmarks/**"
---

# Data and evaluation rules

## Sources
- Each manifest declares `role`: `train`, `tasks_and_train`, or `eval_only`. Never train on `eval_only` sources
  (`benchmarks.yaml`, `mainframe_instruct.yaml`).
- Pin `source.revision` and confirm `license.verified` at download time (currently TODOs).
- Downloads go to `data/raw/` scratch; the real output is a W&B artifact.
- SFT data excludes `.cpy` copybooks (`ext == ".cpy"` in the manifest); they stay in `clean-corpus` as dependencies.
- Dataset plan and source priority (start with real COBOL): [docs/dataset.md](../../docs/dataset.md).
- Row schemas of every artifact: [docs/contracts.md](../../docs/contracts.md).

## No leakage
Benchmarks (`benchmarks/`) are never trained on. Every training set is decontaminated against them before use.
```python
# BAD: build SFT rows straight from clean-corpus / task-pool
rows = to_sft_rows(read_jsonl(task_pool))
# GOOD: SFT reads only the decontaminated alias, produced by `make decontam` (n-gram containment vs COBOLEval etc.)
dataset = run.use_artifact(f"{PROJECT}/sft-dataset:decontaminated-v2")
```
- Held-out CodeNet tasks (`eval.heldout` split) never enter `sft-dataset`; the split is fixed by `data.split_seed` 42.

## Fixed comparison protocol
- Models are compared only on the fixed held-out split and by the eval protocol and criteria of
  [docs/sft-v1-preregistration.md](../../docs/sft-v1-preregistration.md). Do not create new splits or change
  criteria after seeing results.
- Same decoding for base and adapted (non-thinking temperature 0.7, top_p 0.8) unless a config says otherwise.

## Execution
- Execute generated code only inside the `eval-sandbox` container (no network, read-only, resource-capped).
  Vast instances have no Docker, so execution-based eval runs locally for now.
