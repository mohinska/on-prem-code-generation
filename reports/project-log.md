# Project log

Running log of decisions, code changes, runs and results (exact numbers). Source for the reports and the paper.

## Results

### raw-corpus v0 (Stack v1 + X-COBOL, no CodeNet)
- What: `collect` over Stack v1 COBOL and X-COBOL, before any filter.
- Numbers (`stats.json`, identical to artifact metadata): 8203 files, 511,066,885 bytes; xcobol 5195, stack_v1 3008.
- Artifact: `mllab-team/onprem-codegen/raw-corpus:v0` (alias `latest`, 2026-10-11), files `manifest.parquet`, `stats.json`.
- Run: https://wandb.ai/mllab-team/onprem-codegen/runs/5wwbocha, commit `ac99c4b1e327e1a6ab95eaf16f5c360e637bf34d`.
- Note: `data/manifests/{stack_v1_cobol,xcobol}.yaml` still have TODO/null fields in this tree; artifact not re-created.

### cobc flags for the corpus syntax check
- What: `cobc -fsyntax-only` (GnuCOBOL 3.2.0, `eval-sandbox`, no network) on 50 random non-`.cpy` files from raw-corpus (25 stack_v1, 25 xcobol, seed 0), three flag sets.
- Numbers (files that pass): plain 18/50, `-std=ibm` 17/50, `-std=ibm -free` 8/50.
- Failures are mostly fixed-format issues (`invalid indicator X at column 7`), missing copybooks, and dialect keywords.
- Decision: `execute(content, [])` stays as is. No flags parameter on `execute()`: `-std=ibm` does not help and `-free` hurts. The failures are tagged, not fixed.
