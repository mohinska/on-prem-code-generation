# Project log

Decisions, runs and results with exact numbers. Source for the reports and the paper.

## Results

### raw-corpus:v0 (Stack v1 + X-COBOL)
8203 files, 511,066,885 bytes (xcobol 5195, stack_v1 3008). Old source names, the current `clean` can't read it. [Run](https://wandb.ai/mllab-team/onprem-codegen/runs/5wwbocha), commit `ac99c4b`.

### cobc flags for the syntax check
50 random non-`.cpy` files (seed 0), GnuCOBOL 3.2.0: plain 18/50, `-std=ibm` 17/50, `-std=ibm -free` 8/50. Decision: no flags. Failures are tagged, not fixed.

### raw-corpus:v2 and clean-corpus:v4
`make data` on Stack v1, X-COBOL, CodeNet; commit `ddec071` (clean: `0ff425b`, run `clean-0ff425b`, input `raw-corpus:v2`).
- raw-corpus:v2: 8930 files, 511,697,363 bytes (xcobol 5195, stack_v1_cobol 3008, codenet_cobol 727).
- clean-corpus:v4: 4804 kept, 66,044,902 bytes. Decontamination: 0 hits.
- Syntax check of kept (tag only): ok 2446, compile_error 2140, skipped `.cpy` 218.
- Licenses are only tagged. CodeNet licence is `null`, all 727 count as `license_ok=True`; to confirm with C.
- clean-corpus:v3 had the wrong lineage (`raw-corpus:v0`), superseded by v4.

| source | in | exact_dup | near_dup | too_short | not_text | high_nonalnum | too_long | decontam | kept |
|---|---|---|---|---|---|---|---|---|---|
| stack_v1_cobol | 3008 | 0 | 396 | 540 | 14 | 50 | 0 | 0 | 2008 |
| xcobol | 5195 | 1452 | 951 | 346 | 174 | 16 | 1 | 0 | 2255 |
| codenet_cobol | 727 | 40 | 43 | 103 | 0 | 0 | 0 | 0 | 541 |
| total | 8930 | 1492 | 1390 | 989 | 188 | 66 | 1 | 0 | 4804 |

### task-pool:v1 (CodeNet)
`make codenet`, commit `e3c6c9a`. 4053 problems, 3867 with tests, 2104 tasks kept (dropped: has_image 893, non_ascii 862, no_tests 186, too_long 8). Archive sha256 `7e96620307f874269a62cc0225f2a5a13d2de2b3c368cf61319c87ec276620fb`. v0 counted `problem_list.csv` as a problem (4054).

### decontamination
10-gram shingles, threshold 0.5, against COBOLEval (146 tasks) and CobolCodeBench (92). `make decontam` checks an `sft-dataset` version and sets the alias `decontaminated` if there are no hits. Not run on W&B yet: no sft-dataset version exists.

### rejection sampling smoke (20 tasks, no W&B)
Commit `dc9dfc4`, `configs/data/rejection_sampling.yaml` (n 16, temperature 0.7, non-thinking), vLLM 0.32.0 on 1× H100 NVL.
- 320 samples, 1 of 20 tasks solved, 2 sft rows. compile_error 280, runtime_error 13, wrong_output 10, no_program 15, ok 2.
- Cause: 242 of 305 programs read stdin as a file (`ASSIGN TO STDIN`), 227 of those don't compile. `IO_STDIN` in `prompt.py` now asks for ACCEPT and DISPLAY.
- Sandbox check: 38 of 40 accepted CodeNet submissions pass the task-pool tests.
- vLLM needs `--max-num-seqs 256`; the default 1024 fails at startup on Mamba cache blocks.

### still missing
Baseline eval and SFT results (A, C), full rejection-sampling run and `sft-dataset` (C), decontamination of `sft-dataset` (B), CodeNet licence (C), reports.
