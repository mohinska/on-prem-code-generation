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

| Hints | GPU, concurrency | Tasks solved | ok | compile_error | wrong_output | runtime_error | timeout | no_program | Wall time |
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
Commit `f84ab46`, generation_hints above, H200 NVL, concurrency 16, no W&B. `task-pool:v1` is ordered by problem id: AOJ problems first, AtCoder later; 288 of the 295 tasks with a human solution are AtCoder tasks of at most 500 points. The first 20 tasks are among the easiest, so the 5 of 20 above overestimates the pool.

| Tasks | Tasks solved | ok | compile_error | wrong_output | timeout | runtime_error | no_program (truncated) | sft rows |
|---|---|---|---|---|---|---|---|---|
| first 500 (positions 0-499) | 17 of 500 | 44 | 3082 | 324 | 43 | 2 | 4505 (4487) | 29 |
| 40 random of the other 1604 (seed 0) | 5 of 40 | 20 | 418 | 55 | 1 | 0 | 146 (146) | 13 |

- First 500 by position: 0-99 has 39 ok, 100-499 has 5 ok. On hard tasks the model often declares variables one by one (`WS-ITEM80`, `WS-ITEM81`) until `max_tokens`.
- The 40 by AtCoder score: ≤ 300 points 4 of 18 tasks (19 ok), 400-500 points 1 of 9 (1 ok), ≥ 600 points 0 of 8, no score 0 of 5.
- Order of generation: AtCoder tasks up to 500 points (`max_score: 500`), then `--second-pass` (16 more samples) on the unsolved tasks up to 300 points, then all other tasks (`max_score: null`). Human rows use every task.

### rejection sampling: full generation
All 2104 tasks of `task-pool:v1`, generation_hints above, H200 NVL, concurrency 16, no W&B, order as above (commits `bd17060`, `d6d6b37`). Each row is cumulative and counts a program as `ok` after one test.

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
811 rows on 349 tasks (rejection_sampling 328 on 125 tasks, codenet_cobol 483 on 288 tasks; 64 tasks have both). `tasks_solved` 145, dropped: `over_cap` 222, `few_tests` 53, `too_hard` 6. Inputs `task-pool:v1`, `clean-corpus:v4`; generations from H200 NVL, re-run and uploaded locally. [Run](https://wandb.ai/mllab-team/onprem-codegen/runs/7b6m2nvv), commit `6388dd6`.

### sft-dataset:v0 review (C3, numbers)
Tokens counted with the Qwen3.8 tokenizer on prompt + completion (chat template tokens not included).

| Source | Rows | Tasks | Tokens median / p90 / max | Completion median |
|---|---|---|---|---|
| rejection_sampling | 328 | 125 | 566 / 881 / 1784 | 274 |
| codenet_cobol | 483 | 288 | 614 / 1109 / 3620 | 294 |
| all | 811 | 349 | 599 / 1014 / 3620 (p99 1696) | |

- Length histogram (tokens): 0–256: 2, 256–512: 255, 512–768: 342, 768–1024: 134, 1024–1536: 59, 1536–2048: 14, 2048–3072: 4, 3072–4096: 1, over 4096: 0. `max_length` 4096 drops nothing (B1: dropped 0 of 811); 5 rows are over 2048.
- Rows per task: max 8, 155 tasks have one row, 39 tasks have 5 or more (top: `codenet/p03085` 8, `codenet/p04011` 8, `codenet/p02915` 7). No completion is cut off (all end with the closing fence), no duplicate completions.
- Rejection sampling yield: 40 816 samples on 2104 tasks, 584 `ok` (1.43%), 145 tasks solved (6.9%); the rest of the status counts are compile_error 24 027, no_program 11 476, wrong_output 4445.
- Epochs: with 712 train rows B1 used 0.56 of an epoch in 25 steps and eval loss was still falling (0.419 → 0.313). Proposed for `sft-v0`: 3 epochs (≈ 135 steps ≈ 35 min), one checkpoint and eval per epoch (`save_steps` 45), keep the checkpoint with the lowest eval loss.
- Manual review of 20 random rows (10 per source, seed 13): **go**. Thresholds are met (811 rows ≥ 500, 349 tasks ≥ 200). `sft-v0` settings: `sft.yaml` as is (lr 2e-4, r 16, alpha 32, batch 1 × accumulation 16, `max_length` 4096, `eval_fraction` 0.1) with `train.num_train_epochs=3`, `train.save_steps=44` (one checkpoint and eval per epoch), `train.logging_steps=5`.

### sft-v0 (SFT, 2.1)
`--config-name sft run_name=sft-v0 train.num_train_epochs=3 train.save_steps=44 train.logging_steps=5` on `sft-dataset:decontaminated` (v0): QLoRA 4-bit, r 16, alpha 32, dropout 0.05, lr 2e-4 cosine, batch 1 × accumulation 16, `max_length` 4096, 712 train / 99 eval rows (split by task), 135 steps, 1× RTX PRO 6000 Blackwell (98 GB), commit `9c4da41`, started 2026-11-06 22:11 UTC and finished 23:02 UTC. [Run sft-v0](https://wandb.ai/mllab-team/onprem-codegen/runs/sft-v0), run from a one-off queue script (in commit `9c4da41`).

| Step | 44 | 88 | 132 | 135 (end) |
|---|---|---|---|---|
| Eval loss | 0.2170 | 0.1793 | 0.1752 | 0.1752 |
| Eval mean token accuracy | 0.9339 | 0.9438 | 0.9453 | 0.9457 |

- Train loss 0.55 (first logged) → 0.137 (mean of the last 5 logs). Eval loss flattens between epochs 2 and 3 and does not rise, so the last checkpoint is used. Median 15.2 s per step, peak memory 26.2 GB, 116 727 808 trainable parameters.
- Outputs: `sft-v0-adapter:candidate` (v0), `sft-v0-ckpt` v0–v3 (`step-44`, `step-88`, `step-132`, `step-135`), samples table in the run.
- The W&B owner of the run is `irynkasv`: the instance used the W&B key stored in the Vast account environment variables.

### sft-v0 generation (non-thinking, 3.1)
`configs/eval/sft-v0.yaml` (same protocol as the frozen baseline, differs only in `run_name`, `adapter`, `server.model` = `cobol`, `modes` = non-thinking), vLLM 0.32.0 serving `sft-v0-adapter:candidate` as `cobol`, `serve.check` passed (both models answer and the adapter changes the logprobs), commit `9c4da41`. [Run sft-v0-generate](https://wandb.ai/mllab-team/onprem-codegen/runs/o47hz4el), 3 h 7 min, artifact `eval-generations:sft-v0` (v4, 9246 rows).

- Rows per group are identical to the baseline (COBOLEval greedy 438 / sampled 2920, CobolCodeBench completion 138 / 920 and instruct 138 / 920, HumanEval-Python 492 / 3280). `finish_reason` stop 8713, length 533 (5.8%).
- Execution (3.2): local Docker sandbox (GnuCOBOL 3.2.0), [run ke91f26o](https://wandb.ai/mllab-team/onprem-codegen/runs/ke91f26o), `eval-results:sft-v0` (v6), commit `9c4da41`. Full report with the base vs adapted table, CIs and diagnostics: [sft-v0.md](sft-v0.md).

### sft-v0 results (non-thinking, frozen protocol)
pass@1 / pass@10 / compile rate as fractions, Δ pass@1 with the 95% paired bootstrap CI over tasks (2000 resamples, seed 0).

| Group | pass@1 base → sft-v0 | Δ pass@1 [CI] | pass@10 base → sft-v0 | compile base → sft-v0 | truncated base → sft-v0 |
|---|---|---|---|---|---|
| COBOLEval greedy | 0.043 → 0.000 | −0.043 [−0.075, −0.016] | | 0.114 → 0.000 | 0.016 → 0.034 |
| COBOLEval sampled | 0.019 → 0.003 | −0.015 [−0.023, −0.009] | 0.150 → 0.028 | 0.097 → 0.141 | 0.006 → 0.021 |
| CobolCodeBench completion greedy | 0.007 → 0.000 | −0.007 [−0.022, +0.000] | | 0.116 → 0.094 | 0.304 → 0.565 |
| CobolCodeBench completion sampled | 0.008 → 0.020 | +0.012 [+0.000, +0.027] | 0.048 → 0.083 | 0.111 → 0.134 | 0.102 → 0.325 |
| CobolCodeBench instruct greedy | 0.000 → 0.000 | +0.000 [+0.000, +0.000] | | 0.145 → 0.130 | 0.116 → 0.101 |
| CobolCodeBench instruct sampled | 0.001 → 0.000 | −0.001 [−0.003, +0.000] | 0.011 → 0.000 | 0.109 → 0.155 | 0.045 → 0.067 |
| HumanEval-Python greedy | 0.972 → 0.959 | −0.012 [−0.043, +0.018] | | 0.998 → 1.000 | 0.002 → 0.000 |
| HumanEval-Python sampled | 0.969 → 0.934 | −0.034 [−0.057, −0.014] | 0.994 → 0.986 | 0.999 → 0.999 | 0.001 → 0.001 |

- Success criterion (COBOLEval pass@1 above the baseline outside the CI, compile rate up, HumanEval-Python down by at most 2 points): **not met**.
- COBOLEval failures come from two format artifacts, measured locally on the stored generations (no W&B, protocol untouched): the first program line loses its indentation because `build_prompt` strips the task text, and sft-v0 copies it (417 of 423 greedy compile errors are `invalid indicator 'F' at column 7`; baseline answers are indented in 402 of 438 greedy cases, sft-v0 in 0); with the indentation restored the compile rate of sft-v0 is 0.646 greedy / 0.589 sampled (baseline 0.121 / 0.132) but pass@1 stays lower (greedy 0.043 → 0.007, Δ −0.037 [−0.071, −0.005]; sampled 0.030 → 0.013, Δ −0.016 [−0.028, −0.004]; pass@10 0.196 → 0.088, Δ −0.108 [−0.168, −0.050]). 264 of its 271 greedy wrong outputs contain `STOP RUN` (799 of the 811 training programs contain it, none ends a subprogram with `GOBACK`); in COBOLEval the program is called as a subprogram.
- Second diagnostic (indentation restored and `STOP RUN` rewritten to `GOBACK` in the stored answers of both models, local, no W&B, 3358 answers each): COBOLEval greedy pass@1 0.062 → 0.322 (Δ +0.260 [+0.192, +0.326]), sampled pass@1 0.058 → 0.264 (Δ +0.207 [+0.168, +0.249]), sampled pass@10 0.300 → 0.608 (Δ +0.308 [+0.240, +0.373]); greedy `ok` 27 → 141 programs. Post hoc and it edits the model answer, so it explains the frozen result and is not a result of sft-v0; it motivates `sft-v1` (prompt fix, `GOBACK` in the training data).
- The frozen baseline understates the base model on COBOLEval for the same indentation reason (sampled pass@1 0.019 → 0.030 when the first line is indented).
- Unbiased test on the training distribution: not possible with the current `task-pool`; only 19 of the 1755 tasks outside the training set had a passing base sample, so base and adapted cannot be compared fairly on them.

### SFT prompt format vs benchmark format
`sft-dataset:v0` (811 rows) against COBOLEval (146 tasks) and CobolCodeBench (46 + 46). Count of programs / task fields containing the marker (`LINKAGE SECTION`; file I/O = `SELECT … ASSIGN` or `FILE SECTION`; `ACCEPT `; `DISPLAY `):

| Set | LINKAGE | file I/O | ACCEPT | DISPLAY |
|---|---|---|---|---|
| SFT completions, all 811 | 0 (0%) | 51 (6%) | 802 (99%) | 811 (100%) |
| – rejection_sampling (328) | 0 | 0 | 328 | 328 |
| – codenet_cobol (483) | 0 | 51 (11%) | 474 (98%) | 483 |
| COBOLEval prompts (146) | 146 (100%) | 0 | 0 | 0 |
| CobolCodeBench completion (46) | 15 (33%) | 46 (100%) | 1 | 33 |
| CobolCodeBench instruct (46) | 1 (2%) | 46 (100%) | 1 | 33 |

- All 811 SFT prompts use the `IO_STDIN` sentence; COBOLEval uses `IO_COBOLEVAL` (complete a program with a given `LINKAGE SECTION`, answer in `RESULT`), CobolCodeBench uses `IO_FILES`. No SFT row has a `LINKAGE SECTION` and only 51 touch files, so v0 teaches GnuCOBOL syntax and ACCEPT/DISPLAY programs, not the benchmark conventions. Compile rate can rise while pass@1 on the benchmarks does not.
- **Decision: v0 is trained on the data as it is, no benchmark-format rows added.** Building LINKAGE or file-I/O training tasks needs a new task source and a second decontamination against the same benchmarks; v0 is cheap and shows how much transfers. If compile rate rises but COBOLEval / CobolCodeBench pass@1 stays inside the baseline CI, the next step is benchmark-format instruction data (Phase 4, `IO_COBOLEVAL` / `IO_FILES`).

### baseline eval (Qwen/Qwen3.8-27B, no adapter)
vLLM on 1× A100 80 GB, `base`. Pass@1 as fractions; pass@10 only for sampled decoding. Frozen copy of this table, 95% bootstrap CIs per group, the artifact map and the `baseline-final-*` aliases: [baseline.md](baseline.md). `eval.report` adds a paired Δ pass@1 CI to the base vs adapted table and `--ci-of results.jsonl...` prints the single-run CI table (bootstrap over tasks, 2000 resamples, seed 0).

| Benchmark / prompt | Mode | Decoding | pass@1 | pass@10 | compile_rate | no_program | truncation |
|---|---|---|---|---|---|---|---|
| COBOLEval (146 tasks) | non-thinking | greedy | 0.0434 | | 0.1142 | 0.0183 | 0.0160 |
| COBOLEval | non-thinking | sampled | 0.0188 | 0.1496 | 0.0969 | 0.0079 | 0.0058 |
| CobolCodeBench completion (46) | non-thinking | greedy | 0.0072 | | 0.1159 | 0.3043 | 0.3043 |
| CobolCodeBench completion | non-thinking | sampled | 0.0076 | 0.0483 | 0.1109 | 0.1076 | 0.1022 |
| CobolCodeBench instruct (46) | non-thinking | greedy | 0 | | 0.1449 | 0.1159 | 0.1159 |
| CobolCodeBench instruct | non-thinking | sampled | 0.0011 | 0.0109 | 0.1087 | 0.0467 | 0.0446 |
| HumanEval-Python (164) | non-thinking | greedy | 0.9715 | | 0.9980 | 0.0020 | 0.0020 |
| HumanEval-Python | non-thinking | sampled | 0.9686 | 0.9939 | 0.9994 | 0.0006 | 0.0006 |
| COBOLEval | thinking | sampled (n 10) | 0.0514 | 0.2808 | 0.0644 | 0.3753 | 0.3747 |
| HumanEval-Python | thinking | sampled (n 10) | 0.9622 | 0.9756 | 0.9823 | 0.0177 | 0.0171 |

- Headroom confirmed: COBOLEval pass@1 0.0434 (greedy, non-thinking) vs HumanEval-Python 0.9715, COBOLEval compile rate 0.1142. Thinking is better than non-thinking on COBOLEval (pass@10 0.2808 vs 0.1496), held back by truncation.
- Non-thinking: generation [4q4hi07a](https://wandb.ai/mllab-team/onprem-codegen/runs/4q4hi07a) (commit `bb68f18`, 2h 41m), metrics [al1olufk](https://wandb.ai/mllab-team/onprem-codegen/runs/al1olufk) (commit `931124b`), full protocol `configs/eval/baseline.yaml`.
- Thinking is cut down (`configs/eval/baseline-thinking.yaml`, `baseline-thinking-humaneval.yaml`): COBOLEval and HumanEval-Python only (no CobolCodeBench), sampled n 10 (no greedy in the manifest); full thinking on all benchmarks with n 20 would take ~4.5 days on 1× A100 (answers ~21k tokens). COBOLEval generation [ada15z6w](https://wandb.ai/mllab-team/onprem-codegen/runs/ada15z6w) (12h 55m, A100), HumanEval-Python generation [6epckk6y](https://wandb.ai/mllab-team/onprem-codegen/runs/6epckk6y) (commit `6db21f1`, RTX PRO 6000 S, vLLM 0.32.0, 164 requests, 1640 samples, 88 min on the instance after 40 requests were generated from a laptop over an SSH tunnel and copied over).
- Metrics: COBOLEval [p3lre4hl](https://wandb.ai/mllab-team/onprem-codegen/runs/p3lre4hl), HumanEval-Python [eahirhs4](https://wandb.ai/mllab-team/onprem-codegen/runs/eahirhs4), both re-executed after the fix below.
- 37.47% of the COBOLEval thinking answers hit the 28000-token limit without a program (HumanEval-Python 1.71%), so the COBOLEval thinking numbers are a lower bound.
- **Bug found and fixed in the thinking answer parser.** Qwen's chat template puts `<think>` into the prompt, so the reply is `reasoning</think>answer` with no opening tag. `split_reasoning` only removed `<think>...</think>` pairs, so nothing was removed and `program_from_answer` took the first code block, which is in the reasoning. First thinking numbers (before the fix): HumanEval-Python pass@1 0.2713, pass@10 0.9085, compile 0.3281 (1097 of 1640 samples `compile_error`: 1611 replies contained `</think>`, the `reasoning` field was empty in all 1640); COBOLEval pass@1 0.0041, pass@10 0.0411, compile 0.0068 (run mhwq0y9r). Fix: `split_reasoning` splits at the last `</think>` and treats a thinking reply without it as truncated; `eval.run.answer_of` splits old rows of `generations.jsonl` again at execution, so no generation was repeated. Non-thinking numbers are not affected (no `</think>`).

- COBOLEval thinking generation [ada15z6w](https://wandb.ai/mllab-team/onprem-codegen/runs/ada15z6w) has no `git_commit` in W&B.

### B1: SFT dry run on the real dataset
`--config-name sft run_name=sft-b1 train.max_steps=20 train.save_steps=5 train.logging_steps=1` on `sft-dataset:decontaminated` (v0), Qwen3.8-27B QLoRA r 16, `max_length` 4096, batch 1 × gradient accumulation 16, 1× RTX PRO 6000 Blackwell (98 GB), commit `8e531fa`. [Run sft-b1](https://wandb.ai/mllab-team/onprem-codegen/runs/sft-b1).

| | |
|---|---|
| Rows | train 712, eval 99, over-length dropped 0 (≈ 45 steps per epoch) |
| Trainable params | 116 727 808 |
| Peak memory (`perf/peak_mem_gb`) | 23.9 GB, no OOM |
| Seconds per step | first step 80.8 (warm-up), then median 15.3 (14–17; two steps of 34 and 37 around eval) |
| Train loss | 0.608 (step 1) → 0.407 (step 20) |
| Eval loss, steps 5 / 10 / 15 / 20 | 0.4188 / 0.3502 / 0.3214 / 0.3156 |
| Eval mean token accuracy, step 20 | 0.9120 |
| Outputs | `sft-b1-ckpt` v0–v3 (`step-5` … `step-20`), `sft-b1-adapter` (alias `candidate`) |

- **Resume:** the first 20 steps finished before the planned kill (the whole run took ~12 min). Then the local `checkpoints/sft-b1` was deleted and the same command was run with `train.max_steps=25`: it continued the same W&B run `sft-b1` from the `step-20` checkpoint artifact, logged steps 21–25 (15.0–16.9 s each), eval loss at step 25 0.3127, and uploaded `step-25` (`sft-b1-ckpt` v4) and a new `sft-b1-adapter` version. Not tested: a kill in the middle of a step.
- **Estimate for SFT v0 (2.1):** 45 steps per epoch × 15.3 s ≈ 11.5 min per epoch, plus eval and checkpoint uploads (~1–2 min each at `save_steps` 5; use a larger `save_steps` for the real run); 2 epochs ≈ 25–30 min. The planned 1.5 h per epoch assumed longer rows and slower steps.
- Eval loss was still falling at step 25, with 712 train rows only; no sign of overfitting yet.

### Serving the LoRA adapter in vLLM (1.2)
vLLM 0.32.0 (own venv, torch 2.13.0, `--max-num-seqs 256`), model `Qwen3_5ForConditionalGeneration`, 1× RTX PRO 6000 Blackwell (98 GB), `base` + `cobol` = `sft-b1-adapter:candidate` (B1, 25 steps), `scripts/vast_serve.sh`, check `python -m onprem_codegen.serve.check`.

- **Problem found:** with the adapter as trained, both models answered and `serve.check` said `ok`, but `cobol` was identical to `base`: on a fixed 64-token COBOL text the sum of token logprobs was −66.002127 for both (max difference 0.000000), and a 181-token generation matched token by token. The server log had no error, only `Loaded new LoRA adapter: name 'cobol'` and a `_lora_expand_kernel` run. Cause: the text-only trainer saves keys `base_model.model.model.layers.N…`, vLLM serves the multimodal model and looks LoRA weights up under `model.language_model.layers.N…`, so none of the 496 LoRA modules matched (0 of 496 found in the base checkpoint's tensor names) and vLLM applied nothing, silently.
- **Fix:** `serve/adapter_keys.py` renames the keys (992 tensors) after the adapter is fetched; all 496 modules then match the base checkpoint names. Commit `25a9f2f`.
- **After the fix:** same fixed text, sum of logprobs base −66.0021, cobol −62.9725 (max difference 3.2013 over 64 tokens); the generated factorial program differs from base (variable names `WS-N`, `000-MAIN` like the SFT data). Base ↔ adapter toggle by the `model` field works, merging the adapter into the weights is not needed.
- LoRA on the Gated DeltaNet projections (`in_proj_a/b/qkv/z`, `out_proj`) is accepted by vLLM 0.32.0. Warnings about `visual.*` modules being ignored are harmless (text-only adapter).
- Not done: the rename is not applied at training time, only when serving (`train/sft.py` still saves the original keys).

### Held-out task split for sft-v1 (plan 0.2)
`python -m onprem_codegen.data.split_tasks --seed 7 --fraction 0.2` on `task-pool:v1` (file sha256 `74d948eb7687716f619eeef75176308d359f47437425041aa0d01e2b97a643d7`): 421 of 2104 tasks held out, written to `data/splits/heldout_task_ids.json` (seed, fraction and pool hash inside). The choice is random and made before any sft-v1 result exists; these tasks must stay out of rejection sampling and training. Tests: `tests/test_split_tasks.py`.

### sft-v1 success criteria (plan 0.3)
`docs/sft-v1-preregistration.md` committed as `9ca4658` before any sft-v1 training: criteria, verdict rules, held-out split (421 tasks, seed 7), training settings (as sft-v0, seed 42, second seed 43) and comparison protocol. It is not edited after this commit.

### build_prompt keeps the first-line indent (plan 1.1)
`build_prompt` no longer calls `.strip()`: it drops blank lines before the task text and trailing whitespace, and keeps the first line's indentation, so fixed-format COBOL prompts keep their 7 leading spaces. Checked on the stored task texts: all 2104 `task-pool:v1` prompts are byte-identical to the old output (so the 811 `sft-dataset:v0` prompts are too), all 146 COBOLEval and 46 CobolCodeBench completion prompts now start with 7 spaces, and CobolCodeBench instruct (46) and HumanEval-Python (164) are unchanged. Tests: `tests/test_prompt.py`. Baseline v2 on the affected groups is still to be measured (plan 1.4, 2.2).

### sft-dataset:v1-lite (plan 1.2, 1.5)
- 1.2, run [ea6c2jfb](https://wandb.ai/mllab-team/onprem-codegen/runs/ea6c2jfb), `sft-dataset:v1`: from `sft-dataset:v0` (811 rows) the 141 rows of held-out tasks were removed, `STOP RUN` was replaced by `GOBACK` in 662 programs, 8 programs with no terminator at all were dropped, and every remaining program was run again with its tests in the eval-sandbox (generated rows on all samples of the statement, human rows on the task's test): 0 failed. 662 rows on 283 tasks, 100% with `GOBACK`, 0 held-out tasks, 0 benchmark hits.
- 1.5, `sft-dataset:v2` (aliases `v1-lite`, `decontaminated-v1`; `decontaminated` still points to v0): for half of the tasks (seed 7) the row is a completion row, prompt = task text + program up to `DATA DIVISION` + `IO_COMPLETE`, answer = the rest; prefix + answer is the checked program byte for byte. 326 of 662 rows (49.2%), 0 benchmark hits on the new prompts. The first version of the splitter missed `DATA   DIVISION.` with padded spaces and lower case (135 programs, share 39.6%); fixed before the artifact was built.

### baseline v2 generation (ROLE=base)
`ROLE=base bash scripts/sft_v1_queue.sh` on Vast instance 59609324, 1× A100 SXM4 80 GB ($1.12/h, datacenter, image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel`), commit `c29fdf9`, vLLM 0.32.0 serving `base` only (`max_model_len` 32768), W&B owner `dovhai-pn`. Queue 2026-11-07 18:53 → 23:58 UTC; the instance was destroyed 2026-11-08 00:04 UTC after both artifacts were `COMMITTED` in W&B.

| Step | Time (UTC) | Duration |
|---|---|---|
| benchmarks ([run 80khgzxr](https://wandb.ai/mllab-team/onprem-codegen/runs/80khgzxr)) | 18:53:03 → 18:53:09 | 6 s |
| task-pool | 18:53:09 → 18:53:14 | 5 s |
| vllm-env (second start, see below) | 18:53:59 → 18:55:02 | 1 min |
| serve | 18:55:02 → 19:00:22 | 5 min 20 s |
| check (`base ok`) | 19:00:22 → 19:00:24 | 2 s |
| generate-baseline-v2 ([run whdrytj0](https://wandb.ai/mllab-team/onprem-codegen/runs/whdrytj0)) | 19:00:24 → 20:39:53 | 1 h 39 min |
| heldout-base ([run 3qmyo85n](https://wandb.ai/mllab-team/onprem-codegen/runs/3qmyo85n)) | 20:39:53 → 23:58:11 | 3 h 18 min |

- `eval-generations:baseline-v2` (v6): 768 requests, 0 failed, 4416 rows, all non-thinking: COBOLEval greedy 438 / sampled 2920, CobolCodeBench completion greedy 138 / sampled 920. `finish_reason` length 145 (3.3%): COBOLEval 4 / 15, CobolCodeBench completion 31 / 95. 0 rows contain `</think>`.
- `eval-generations:heldout-base` (v9): 842 requests, 0 failed, 2526 rows on 421 held-out tasks (greedy 421, sampled 2105). `finish_reason` length 653 (25.9%): greedy 145, sampled 508. 0 rows contain `</think>`.
- Instance setup on a fresh Vast instance: the Vast account environment variables override `--env` of `vastai create instance`, so the instance got the account `WANDB_API_KEY` (owner `irynkasv`). The own key was copied to `/root/.wandb_env` and sourced in the shell. The account has no `GITHUB_TOKEN`, so it was passed with `--env` (the repo is private). The variables are not visible in an SSH shell, only in `/proc/1/environ`, so they were exported from there before the queue started.
- The first `vllm-env` start failed: `pip install uv` hit `externally-managed-environment` (PEP 668), because `vast_onstart.sh` sets `PIP_BREAK_SYSTEM_PACKAGES=1` only for itself. With `export PIP_BREAK_SYSTEM_PACKAGES=1` in the shell the same command skipped the done steps and continued.

### sft-v1 (SFT, plan 2.1, 2.3) and its generations
`ROLE=adapter bash scripts/sft_v1_queue.sh` on one Vast instance (1× RTX PRO 6000 Blackwell), commit `c29fdf9`. Both runs are `sft-v0` settings (QLoRA 4-bit, r 16, alpha 32, lr 2e-4, 3 epochs, 111 steps) on `sft-dataset:decontaminated-v1` (v2), 585 train / 77 eval rows with the same split for both seeds (`data.split_seed` 42).

| Run | Seed | Wall time | Eval loss, step 37 / 74 / 111 | Eval token accuracy, step 111 | Train loss | Adapter |
|---|---|---|---|---|---|---|
| [sft-v1](https://wandb.ai/mllab-team/onprem-codegen/runs/sft-v1) | 42 | 44 min (15:06–15:50 UTC) | 0.1891 / 0.1618 / 0.1588 | 0.9521 | 0.2302 | `sft-v1-adapter:candidate` (v0) |
| [sft-v1-seed2](https://wandb.ai/mllab-team/onprem-codegen/runs/sft-v1-seed2) | 43 | 32 min (15:51–16:23 UTC) | 0.1913 / 0.1619 / 0.1589 | 0.9518 | 0.2353 | `sft-v1-seed2-adapter:candidate` (v0) |

- Eval loss still falls slightly at epoch 3 and does not rise; the last checkpoint is used. The two seeds differ by ≤ 0.002 in eval loss.
- vLLM 0.32.0 served both adapters (`cobol`, `cobol-seed2`); `serve.check` passed, max logprob difference vs base 5.01 (`cobol`) and 5.35 (`cobol-seed2`).
- Generations (`configs/eval/sft-v1.yaml`, `sft-v1-seed2.yaml`; held-out: 421 tasks × greedy + sampled = 842 requests per model, 0 failed):

| Artifact `eval-generations:` | Version | Run | Size, time |
|---|---|---|---|
| `sft-v1` | v5 | [rw0tmnvp](https://wandb.ai/mllab-team/onprem-codegen/runs/rw0tmnvp) | 9246 rows, all groups; 2 h 42 min |
| `heldout-sft-v1` | v7 | [18qxqhih](https://wandb.ai/mllab-team/onprem-codegen/runs/18qxqhih) | 842 requests; 2 h 47 min (19:15–22:02 UTC) |
| `sft-v1-seed2` | v8 | [eo027095](https://wandb.ai/mllab-team/onprem-codegen/runs/eo027095) | 2920 rows, COBOLEval sampled only (by design); 35 min |
| `heldout-sft-v1-seed2` | v10 | [up5xqy99](https://wandb.ai/mllab-team/onprem-codegen/runs/up5xqy99) | 842 requests; ≈ 3 h 15 min (to 01:52 UTC) |

- Whole queue: 15:06–01:52 UTC (≈ 10 h 45 min); generation took ≈ 9 h of it. Truncated (`finish_reason` length): 281 of 9246 rows (sft-v1), 91 of 2920 (seed 2). The instance was destroyed by hand after the queue (AUTODESTROY not used).
- The W&B owner of the runs is `irynkasv` (W&B key from the Vast account environment variables).
- Execution in the sandbox and the verdict: see "sft-v1 verdict" below.

### held-out base execution (sandbox)
`heldout execute --models base` on `eval-generations:heldout-base` (v9), locally in `eval-sandbox` (Docker Desktop, Windows), pool `task-pool:v1`, 421 tasks, 2526 rows. No `</think>` in programs.

| Decoding | Rows | ok | wrong_output | runtime_error | timeout | compile_error | no_program |
|---|---|---|---|---|---|---|---|
| greedy | 421 | 2 | 21 | 2 | 0 | 251 | 145 |
| sampled (5 per task) | 2105 | 7 | 78 | 6 | 5 | 1505 | 504 |

- A container killed for its memory limit (`docker run` exit 137) counts as the program's `runtime_error` (`compile_error` if cobc left no binary), not as a sandbox failure. Before, one program (`codenet/p03971` greedy, 100 000-char records) stopped the whole run at 2525 of 2526. Test: `test_container_killed_for_memory_is_a_program_failure`.

### sft-v1 verdict (plan 3.2)
Execution in `eval-sandbox` at commit `c63c99b`: [baseline-v2](https://wandb.ai/mllab-team/onprem-codegen/runs/7idjdqq0) (`eval-results:baseline-v2`, v7, 4416 rows), [sft-v1](https://wandb.ai/mllab-team/onprem-codegen/runs/9iif0lhu) (`eval-results:sft-v1`, v8, 9246 rows), [sft-v1-seed2](https://wandb.ai/mllab-team/onprem-codegen/runs/jeodg2bm) (`eval-results:sft-v1-seed2`, v9, 2920 rows); held-out locally (2526 rows per model, `report_cobol.json`, `report_cobol-seed2.json`). No `</think>` in any program. Verdict by `eval.verdict` (commit `c29fdf9`). Report: [sft-v1.md](sft-v1.md).

**Вердикт: частковий, нестабільний** (seed 42 and seed 43 differ on the main criterion).

| Criterion (seed 42) | Met | Numbers |
|---|---|---|
| Main: COBOLEval sampled pass@1, compile rate | no | 0.021 → 0.021, Δ +0.000 [−0.011, +0.011]; compile 0.086 → 0.059 |
| Extra: COBOLEval pass@10, greedy pass@1 | no | pass@10 Δ +0.006 [−0.051, +0.068]; greedy Δ −0.034 |
| Guard Python | no | HumanEval-Python sampled pass@1 0.969 → 0.919 (Δ −0.050) |
| Guard CobolCodeBench completion | no | Δ pass@1 −0.009 [−0.018, −0.001]; truncation 0.103 → 0.126 |
| Training works (held-out) | yes | sampled pass@1 0.003 → 0.068, Δ +0.065 [+0.048, +0.083] |
| Seed 43: main | yes | 0.021 → 0.055, Δ +0.035 [+0.017, +0.053]; compile 0.086 → 0.137 |
| Seed 43: training works | yes | 0.003 → 0.069, Δ +0.066 [+0.048, +0.085] |

- Exploratory: seed 42 writes the first program line in column 1 in 383 of 429 greedy COBOLEval programs (`invalid indicator`, 380 compile errors); seed 43 mostly keeps the indent. Cause not investigated.
- `sft-v1-adapter:candidate` is not promoted; README "Run the trained model" unchanged.

### still missing
SFT results (A, C), thinking baseline on CobolCodeBench (not run, see above), CodeNet licence (C), reports.
