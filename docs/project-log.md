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
10-gram shingles, threshold 0.5, against COBOLEval (146 tasks) and CobolCodeBench (92). `make decontam` checks an `sft-dataset` version and sets the alias `decontaminated` if there are no hits. `sft-dataset:v0`: 811 rows checked, 0 hits. [Run](https://wandb.ai/mllab-team/onprem-codegen/runs/jh9xasa1).

### rejection sampling smoke (20 tasks, no W&B)
Commit `dc9dfc4`, `configs/data/rejection_sampling.yaml` (n 16, temperature 0.7, non-thinking), vLLM 0.32.0 on 1× H100 NVL.
- 320 samples, 1 of 20 tasks solved, 2 sft rows. compile_error 280, runtime_error 13, wrong_output 10, no_program 15, ok 2.
- Cause: 242 of 305 programs read stdin as a file (`ASSIGN TO STDIN`), 227 of those don't compile. `IO_STDIN` in `prompt.py` now asks for ACCEPT and DISPLAY.
- Sandbox check: 38 of 40 accepted CodeNet submissions pass the task-pool tests.
- vLLM needs `--max-num-seqs 256`; the default 1024 fails at startup on Mamba cache blocks.

### rejection sampling: generation hints (same 20 tasks, no W&B)
Base `931124b` plus `generation_hints` in `configs/data/rejection_sampling.yaml`: appended to the request only, the `sft-dataset` prompt stays `build_prompt` (0 of 15 rows contain a hint). n 16, temperature 0.7, top_p 0.8, max_tokens 2048, non-thinking, vLLM `--max-model-len 8192 --max-num-seqs 256`. Each hint set is cumulative.

| Hints | GPU (Vast instance), concurrency | Tasks solved | ok | compile_error | wrong_output | runtime_error | timeout | no_program | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| none | RTX PRO 6000 S (59220542), 8 | 1 | 1 | 263 | 28 | 5 | 2 | 21 | 4:10 |
| WORKING-STORAGE spelling, `WS-` names, columns 8-72 | RTX PRO 6000 S (59220542), 8 | 3 | 6 | 242 | 43 | 3 | 5 | 21 | 4:11 |
| + no SELECT, FD or files | RTX PRO 6000 S (59220542), 8 | 2 | 4 | 228 | 59 | 0 | 10 | 19 | 4:10 |
| + numbers without leading zeros, spaces or plus signs (`PIC Z(17)9`, `FUNCTION TRIM`) | H200 NVL (59224690), 16 | 5 | 16 | 226 | 47 | 0 | 7 | 24 | 1:42 |

- All 320 samples per run. The C1 gate (≥ 4 of 20 tasks, compile_error clearly below 280) passes with the last set: 15 sft rows from 5 tasks, 0 hardcoded, 0 programs with files.
- Without hints: 39 programs write `WORKING-STORE`, 26 fail first on a reserved data name (`NUMBERS`, `COUNT`). With the first set: 0 `WORKING-STORE`, but 105 of 299 programs use files, 100 of them don't compile and none pass.
- 59 wrong_output of the third set re-run in `eval-sandbox`: 31 wrong values, 14 empty output, 7 different token count, 7 format only (leading zeros, spaces).
- Chat template ruled out: the prompt rendered by the server is identical to `apply_chat_template` of `Qwen/Qwen3.8-27B` with `enable_thinking=False`, and `train/sft.py` uses the same.

### rejection sampling: yield by task difficulty
Commit `f84ab46`, generation_hints above, H200 NVL (Vast 59224690), concurrency 16, no W&B. `task-pool:v1` is ordered by problem id: AOJ problems first, AtCoder later; 288 of the 295 tasks with a human solution are AtCoder tasks of at most 500 points. The first 20 tasks are among the easiest, so the 5 of 20 above overestimates the pool.

| Tasks | Tasks solved | ok | compile_error | wrong_output | timeout | runtime_error | no_program (truncated) | sft rows |
|---|---|---|---|---|---|---|---|---|
| first 500 (positions 0-499) | 17 of 500 | 44 | 3082 | 324 | 43 | 2 | 4505 (4487) | 29 |
| 40 random of the other 1604 (seed 0) | 5 of 40 | 20 | 418 | 55 | 1 | 0 | 146 (146) | 13 |

- First 500 by position: 0-99 has 39 ok, 100-499 has 5 ok. On hard tasks the model often declares variables one by one (`WS-ITEM80`, `WS-ITEM81`) until `max_tokens`.
- The 40 by AtCoder score: ≤ 300 points 4 of 18 tasks (19 ok), 400-500 points 1 of 9 (1 ok), ≥ 600 points 0 of 8, no score 0 of 5.
- Order of generation: AtCoder tasks up to 500 points (`max_score: 500`), then `--second-pass` (16 more samples) on the unsolved tasks up to 300 points, then all other tasks (`max_score: null`). Human rows use every task.

### rejection sampling: full generation
All 2104 tasks of `task-pool:v1`, generation_hints above, H200 NVL (Vast 59224690), concurrency 16, no W&B, order as above (commits `bd17060`, `d6d6b37`). Each row is cumulative and counts a program as `ok` after one test.

| After | Tasks sampled | Samples | Tasks solved | ok | compile_error | wrong_output | timeout | runtime_error | no_program (truncated) | sft rows |
|---|---|---|---|---|---|---|---|---|---|---|
| tasks up to 500 points (842 more) | 1382 | 22112 | 192 | 730 | 12607 | 2194 | 134 | 13 | 6434 (6421) | 492 |
| `--second-pass` (447 tasks, 16 more samples) | 1382 | 29264 | 215 | 764 | 17798 | 3428 | 186 | 17 | 7071 (7059) | 525 |
| all other tasks (722 more) | 2104 | 40816 | 250 | 867 | 24027 | 4167 | 244 | 35 | 11476 (11471) | 593 |

### rejection sampling: one test is not enough
Every task in `task-pool:v1` has one test, the first sample of its statement. The other 722 tasks (428 of them at least 600 points) gave 35 "solved" tasks, including a 2400-point task solved by `IF N > 100` print -1, else 1. So all `ok` programs were re-run on the other samples of the statement, in `eval-sandbox`. `all_tests` uses them only if parsing reproduces the task's own test; 814 of the 867 programs (233 of 250 tasks) got at least one more test.

| Task score | Programs re-run | Pass all samples | Tasks before | Tasks after |
|---|---|---|---|---|
| ≤ 300 | 682 | 481 | 184 | 116 |
| 400-500 | 24 | 5 | 11 | 3 |
| ≥ 600 | 39 | 6 | 23 | 3 |
| no score | 69 | 39 | 15 | 6 |
| total | 814 | 531 | 233 | 128 |

- 283 of 814 (34.8%) passed one test by chance. Of the tasks solved by the last two generation steps, 12 of 23 (second pass) and 7 of 35 (all other tasks) remain.
- Control: 520 accepted CodeNet submissions on the same samples. 407 pass the sandbox comparison; most of the rest print `   42` for 42, which the judge accepts. Ignoring spacing, 497 of 520 pass (12 wrong_output, 3 of them on tasks accepting any valid answer; 10 compile_error; 1 timeout). All 283 generated failures also fail when ignoring spacing.
- Rules: generated rows need at least 2 passed tests (`keep.min_tests`, drops 53 programs on 17 tasks) and at most 500 points (`keep.max_score`, drops 6 rows on 3 tasks; read by hand they are too-simple rules or brute force). Human rows only need to work with our cobc on the task's test, ignoring spacing, because the judge already checked the logic.
- `samples.jsonl` keeps the stronger status (`ok` 867 -> 584, `wrong_output` 4167 -> 4445, `timeout` 244 -> 249), so DPO pairs don't use these programs as chosen.

### sft-dataset:v0
811 rows on 349 tasks (rejection_sampling 328 on 125 tasks, codenet_cobol 483 on 288 tasks; 64 tasks have both). `tasks_solved` 145, dropped: `over_cap` 222, `few_tests` 53, `too_hard` 6. Inputs `task-pool:v1`, `clean-corpus:v4`; generations from Vast 59224690, re-run and uploaded locally. [Run](https://wandb.ai/mllab-team/onprem-codegen/runs/7b6m2nvv), commit `6388dd6`.

### still missing
Baseline eval and SFT results (A, C), CodeNet licence (C), reports.
