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

### clean-corpus v0 with syntax check (local, not uploaded to W&B)
- What: `python -m onprem_codegen.data.clean --no-wandb` on the local raw-corpus v0, commit `5e31624`. Syntax check = `execute(content, [])` (`cobc -fsyntax-only`, no flags, see the cobc flags entry) on kept non-`.cpy` files.
- Numbers (`stats.json`): 8203 in, 4263 kept (65,527,424 bytes; xcobol 2255, stack_v1 2008). Dropped: exact_dup 1452, near_dup 1347, too_short 886, not_text 188, high_nonalnum 66, too_long 1.
- Syntax check of the 4263 kept: ok 1916, compile_error 2129, skipped 218 (the `.cpy` copybooks). Files are only tagged, none dropped.
- Licenses are only tagged (decision): kept includes GPL-3.0 16 files, `kept_by_license_ok` is in `stats.json`.

### decontamination against COBOLEval and CobolCodeBench
- What: `onprem_codegen.data.decontaminate` loads tasks via `eval.benchmarks.load` for the benchmarks in `configs/data/decontam.yaml` (coboleval 146 tasks, cobolcodebench 92 tasks in both modes). Every part (prompt, canonical solution, caller programs, expected outputs) is shingled (10-gram, threshold 0.5) and checked separately; parts shorter than one n-gram are skipped. `clean` writes `by_benchmark` to `stats.json`.
- CLI: `python -m onprem_codegen.data.decontaminate --input <jsonl> --report <path>` for `corpus.jsonl` and `sft.jsonl`, exit code 1 on any hit.
- Numbers: `data/raw/clean-corpus/corpus.jsonl` 4263 checked, 0 hits; `tests/fixtures/sft_smoke.jsonl` 8 checked, 0 hits.
- Decision: the `PROGRAM-ID` entry-point check is not brought back, and HumanEval-Python is not checked.
