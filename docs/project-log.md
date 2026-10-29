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

### alias `sft-dataset:decontaminated`
- `make decontam` (`decontaminate --artifact sft-dataset:latest`) downloads a sft-dataset version, checks `sft.jsonl` with the decontamination above, logs a `decontam-report` artifact and adds the alias `decontaminated` to that version only when there are no hits (the alias then moves to it). `configs/train/sft.yaml` pulls this alias. Not run against W&B yet: there is no sft-dataset version, and C should confirm the flow.

### one set of source names, `collect` without optional sources
- `ManifestRow.source` now uses the contract names `stack_v1_cobol | xcobol | codenet_cobol` (before: `stack_v1`, `codenet`), the same as `source` in the `corpus.jsonl` records; `docs/contracts.md` and C's code are unchanged. `dedup.PRIORITY` follows. The existing local `raw-corpus` v0 manifest has the old names, so it needs a re-run of `collect` before `clean`.
- `collect` has no `if codenet_root`: all three loaders have the same call shape, default roots are `data/raw/<source>/`, missing data stops with a message, `--skip <source>` leaves a source out on purpose.

### raw-corpus:v2 and clean-corpus:v3 with CodeNet
- What: `make data` (collect + clean) on Stack v1, X-COBOL and CodeNet, commit `ddec071`. X-COBOL was re-downloaded from Zenodo (md5 `cbd974ca…` matches, 5195 files) because `data/raw/xcobol/COBOL_Files` was missing; `load_xcobol` now fails if it is missing.
- raw-corpus:v2 (`mllab-team/onprem-codegen`, alias `latest`): 8930 files, 511,697,363 bytes; xcobol 5195, stack_v1_cobol 3008, codenet_cobol 727. Source names are the contract names, so v0 (old names) cannot be read by the current `clean`.
- clean-corpus:v3: 4804 kept of 8930 (66,044,902 bytes). Dropped: exact_dup 1492, near_dup 1390, too_short 989, not_text 188, high_nonalnum 66, too_long 1. kept_by_source: xcobol 2255, stack_v1_cobol 2008, codenet_cobol 541. Syntax check of kept: ok 2446, compile_error 2140, skipped (.cpy) 218. Decontamination hits: 0 (COBOLEval and CobolCodeBench).
- CodeNet: 727 accepted COBOL files, 541 kept (too_short 103, near_dup 43, exact_dup 40); kept ones: ok 530, compile_error 11. Licence is `null` in the config, so all 727 count as `license_ok=True` (no licence found); to be confirmed with C.
- Problems: the first `collect` upload (run `glnvdwp8`) never became a version, probably lost when raw-corpus:v1 (the first, incomplete upload without X-COBOL) was deleted while it was in flight. The local output was logged again as raw-corpus:v2. clean-corpus:v3 and decontam-report:v1 record `raw-corpus:v0` as their input, so their lineage is wrong.
- CodeNet metadata also holds `problem_list.csv`, which matched the `p*.csv` glob; the old task-pool:v0 counted it as a problem (`problems` 4054, `no_tests` 187, one too many each). task-pool:v1 has the right counts.

### task-pool:v1 (CodeNet problems with I/O tests)
- What: `make codenet` on `Project_CodeNet.tar.gz` (7.8 GB, streamed, only the COBOL parts, metadata, problem statements and I/O data extracted), commit `e3c6c9a`.
- Numbers (`stats.json`): 4053 problems, 3867 with tests, 2104 tasks kept. Dropped: has_image 893, non_ascii 862, no_tests 186, too_long 8. One test per task.
- Archive sha256 `7e96620307f874269a62cc0225f2a5a13d2de2b3c368cf61319c87ec276620fb`, now pinned in `configs/data/codenet_cobol.yaml`.
- Artifact: `mllab-team/onprem-codegen/task-pool:v1` (alias `latest`); v0 had 4054 problems and 187 `no_tests` (see the note above).
- The `execute()` contract change (`tests=[]` compiles only) is already in `main` and `origin/main` (merge of `refactor/ponytail-cuts`), so no separate PR is needed.
