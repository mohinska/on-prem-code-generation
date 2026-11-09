# sft-v1: second SFT of Qwen3.8-27B on COBOL, judged by the preregistered criteria

## 1. Summary

| Question | Answer |
|---|---|
| Verdict (criteria fixed before training in [sft-v1-preregistration.md](sft-v1-preregistration.md)) | **Частковий, нестабільний результат.** «Навчання працює» is met, the main criterion is not met for seed 42; seed 43 meets both, so the two seeds give different verdicts. |
| Does training help on unseen tasks of the training distribution? | **Yes, for both seeds.** Held-out CodeNet tasks (421, never trained on): sampled pass@1 0.003 → 0.068 (seed 42) and 0.069 (seed 43), Δ +0.065 [+0.048, +0.083] and +0.066 [+0.048, +0.085]. |
| Does it help on COBOLEval? | **Not for seed 42** (sampled pass@1 0.021 → 0.021, Δ +0.000 [−0.011, +0.011]; greedy 0.034 → 0.000; compile rate 0.086 → 0.059). **Yes for seed 43** (sampled pass@1 0.021 → 0.055, Δ +0.035 [+0.017, +0.053]; compile rate 0.086 → 0.137). |
| Guards | Both fail for seed 42: HumanEval-Python sampled pass@1 0.969 → 0.919 (Δ −0.050, limit −0.02); CobolCodeBench completion pass@1 0.009 → 0.000, Δ −0.009 [−0.018, −0.001]. Guards were not measured for seed 43 (by design, COBOLEval sampled only). |
| Decision | Do not ship `sft-v1-adapter:candidate`; README stays on the current model (step 3.3 applies only to «успіх»). |

## 2. Setup

- **Training:** as `sft-v0` (QLoRA 4-bit, r 16, alpha 32, lr 2e-4, 3 epochs, 111 steps) on `sft-dataset:decontaminated-v1` (v2, from `sft-dataset:v1-lite`: no held-out tasks, `STOP RUN` → `GOBACK`, half the rows in completion format). Seeds 42 (`sft-v1`) and 43 (`sft-v1-seed2`), same data split. Training details: [project-log](project-log.md#sft-v1-sft-plan-21-23-and-its-generations).
- **Comparison:** `sft-v1` against baseline v2 (unmodified model, prompt with the first-line indent kept, plan 1.1) for COBOLEval and CobolCodeBench completion; against the frozen baseline (`eval-results:baseline-final-non-thinking`) for HumanEval-Python and CobolCodeBench instruct, whose prompts did not change. Non-thinking, greedy (3 per task) and sampled (20 per task); pass@k by the unbiased estimator, Δ by paired bootstrap over tasks (2000 resamples, 95% interval).
- **Execution:** GnuCOBOL 3.2 in `eval-sandbox`, commit `c63c99b`. Runs: [baseline-v2](https://wandb.ai/mllab-team/onprem-codegen/runs/7idjdqq0) (`eval-results:baseline-v2`, v7), [sft-v1](https://wandb.ai/mllab-team/onprem-codegen/runs/9iif0lhu) (`eval-results:sft-v1`, v8), [sft-v1-seed2](https://wandb.ai/mllab-team/onprem-codegen/runs/jeodg2bm) (`eval-results:sft-v1-seed2`, v9). Held-out: `python -m onprem_codegen.eval.heldout execute`, local, on `eval-generations:heldout-base` / `heldout-sft-v1` / `heldout-sft-v1-seed2`.
- **Verdict:** `python -m onprem_codegen.eval.verdict` (commit `c29fdf9`), output `outputs/eval/sft-v1-verdict.md` and `verdict.json`.

## 3. Results: baseline v2 vs sft-v1 (seed 42)

| Group | pass@1 base | pass@1 sft-v1 | Δ pass@1 [95% CI] | pass@10 base | pass@10 sft-v1 | Δ pass@10 [95% CI] | compile base | compile sft-v1 | truncated base | truncated sft-v1 |
|---|---|---|---|---|---|---|---|---|---|---|
| coboleval completion/greedy | 0.034 | 0.000 | −0.034 [−0.064, −0.009] | - | - | - | 0.132 | 0.002 | 0.009 | 0.021 |
| coboleval completion/sampled | 0.021 | 0.021 | +0.000 [−0.011, +0.011] | 0.139 | 0.145 | +0.006 [−0.051, +0.068] | 0.086 | 0.059 | 0.005 | 0.009 |
| cobolcodebench completion/greedy | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | - | - | - | 0.072 | 0.000 | 0.225 | 0.399 |
| cobolcodebench completion/sampled | 0.009 | 0.000 | −0.009 [−0.018, −0.001] | 0.064 | 0.000 | −0.064 [−0.127, −0.011] | 0.087 | 0.004 | 0.103 | 0.126 |
| cobolcodebench instruct/greedy ¹ | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | - | - | - | 0.145 | 0.116 | 0.116 | 0.072 |
| cobolcodebench instruct/sampled ¹ | 0.001 | 0.001 | +0.000 [−0.003, +0.003] | 0.011 | 0.011 | +0.000 [−0.033, +0.033] | 0.109 | 0.147 | 0.045 | 0.067 |
| humaneval_python completion/greedy ¹ | 0.972 | 0.925 | −0.047 [−0.083, −0.012] | - | - | - | 0.998 | 0.998 | 0.002 | 0.002 |
| humaneval_python completion/sampled ¹ | 0.969 | 0.919 | −0.050 [−0.076, −0.026] | 0.994 | 0.979 | −0.015 [−0.035, −0.001] | 0.999 | 0.999 | 0.001 | 0.001 |

¹ base = frozen baseline (prompts unchanged since then).

## 4. Criteria

Conditions copied from the preregistration; numbers from `verdict.json` (seed 42, sampled unless the criterion names greedy).

| Критерій | Умова виконання | Виконано | Числа |
|---|---|---|---|
| **Основний** (COBOLEval, sampled n 20) | Δ pass@1 > 0 з нижньою межею інтервалу > 0 **і** вищий compile rate | ні | pass@1 0.021 → 0.021, Δ +0.000 [−0.011, +0.011]; compile 0.086 → 0.059 |
| **Додатковий** | COBOLEval pass@10: Δ > 0 з нижньою межею > 0; greedy pass@1 не нижчий за baseline v2 | ні | pass@10 Δ +0.006 [−0.051, +0.068]; greedy pass@1 Δ −0.034 |
| **Guard Python** | HumanEval-Python sampled pass@1 падає не більше ніж на 0.02 (точкова оцінка) | ні | pass@1 0.969 → 0.919 (Δ −0.050) |
| **Guard CobolCodeBench completion** | pass@1 не нижчий за baseline v2 за інтервалом **і** частка обрізаних відповідей не вища за baseline v2 більш ніж на 10 пунктів | ні | Δ pass@1 −0.009 [−0.018, −0.001]; обрізання 0.103 → 0.126 (+0.023) |
| **«Навчання працює»** (утримані 20% задач CodeNet, stdin) | pass@1 адаптера > pass@1 бази, нижня межа інтервалу > 0 | так | Δ pass@1 +0.065 [+0.048, +0.083] |

**Вердикт: частковий** («навчання працює» виконано, основний ні). Seed 43 meets the main criterion, so the result is marked **нестабільний** (section 6).

## 5. Held-out tasks

421 CodeNet tasks held out before training (`data/splits/heldout_task_ids.json`), stdin/stdout, all sample tests of the statement; greedy (1 per task) and sampled (5 per task). Paired bootstrap over tasks.

| Model | greedy pass@1 | sampled pass@1 | sampled pass@5 | Δ sampled pass@1 [95% CI] | Δ sampled pass@5 [95% CI] | compile (greedy / sampled) | no program (greedy / sampled) |
|---|---|---|---|---|---|---|---|
| base | 0.005 | 0.003 | 0.012 | | | 0.059 / 0.046 | 0.344 / 0.239 |
| sft-v1 (seed 42) | 0.057 | 0.068 | 0.157 | +0.065 [+0.048, +0.083] | +0.145 [+0.112, +0.181] | 0.375 / 0.368 | 0.240 / 0.190 |
| sft-v1-seed2 (seed 43) | 0.074 | 0.069 | 0.154 | +0.066 [+0.048, +0.085] | +0.143 [+0.109, +0.178] | 0.340 / 0.348 | 0.363 / 0.246 |

Both seeds solve about 20 times as many held-out tasks as the base model and agree within 0.01. "No program" is almost always a truncated answer (finish reason `length`).

## 6. Spread of the two seeds

| COBOLEval completion/sampled | base | seed 42 | seed 43 |
|---|---|---|---|
| pass@1 | 0.021 | 0.021 (Δ +0.000 [−0.011, +0.011]) | 0.055 (Δ +0.035 [+0.017, +0.053]) |
| pass@10 | 0.139 | 0.145 (Δ +0.006 [−0.051, +0.068]) | 0.243 (Δ +0.104 [+0.042, +0.171]) |
| compile rate | 0.086 | 0.059 | 0.137 |
| truncated | 0.005 | 0.009 | 0.031 |
| «Навчання працює», Δ held-out pass@1 | | +0.065 [+0.048, +0.083] | +0.066 [+0.048, +0.085] |

The seeds agree on the held-out tasks and on eval loss (≤ 0.002 apart), but not on COBOLEval.

**Exploratory, not part of the verdict:** where COBOLEval compile errors come from. Seed 42 starts the first line of the program in column 1 (`IDENTIFICATION DIVISION.` without the 7-space indent) in 383 of 429 greedy and 2225 of 2893 sampled programs, which fixed-format COBOL rejects (`invalid indicator`, 380 of the greedy compile errors). Seed 43 does so in 326 of 2828 sampled programs; its most frequent compile error is `PROGRAM-ID header missing` (1893). Baseline v2: 36 of 431 greedy, 669 of 2901 sampled. This is the same symptom as artifact 1 of `sft-v0`, now in the model's output although the prompt keeps the indent; no edited answers are counted anywhere in this report.

**What went wrong (diagnosis after the verdict).** Both COBOLEval compile errors come from the training data, not from the prompt or the model:

| Error | Cause in `sft-dataset:decontaminated-v1` | Mostly in |
|---|---|---|
| First line from column 1 (`invalid indicator`) | 343 of 670 programs are human CodeNet solutions in free format (code from column 1). The sandbox compiles stdin tasks with `cobc -x`, which detects free format, so they passed; COBOLEval compiles with `-fformat=variable`, where column 7 is the indicator. The model mixes the two styles. | seed 42 |
| `PROGRAM-ID header missing` | Completion rows (half the data) answer with the remaining lines only; COBOLEval and CobolCodeBench compile the answer alone, without the prompt. | seed 43 |

Each seed picked up a different defect, which explains the spread. Fix for `sft-v2`: free-format programs moved to fixed format and rechecked, completion rows answer with the whole program (`sft-dataset:decontaminated-v2`, [project-log](project-log.md#sft-dataset-v2-fixed-format-whole-program-completion-rows)). The Python drop is not explained by this.

**Diagnostic check (answers edited, not a result).** The stored answers of base and both seeds repaired by the same two rules and executed again ([project-log](project-log.md#sft-v1-format-repair-diagnostic-answers-edited)). COBOLEval sampled pass@1, base → adapted, both repaired: seed 42 0.032 → 0.274 (Δ +0.242 [+0.191, +0.293]), seed 43 0.032 → 0.261 (Δ +0.229 [+0.183, +0.275]); compile 0.123 → 0.654 / 0.629. With the format right the seeds agree; CobolCodeBench completion shows no difference. It does not change the verdict.

## 7. Comparison with sft-v0

Different baselines (sft-v0: frozen baseline; sft-v1: baseline v2 for COBOLEval and CobolCodeBench completion), so only the direction is compared.

| | sft-v0 | sft-v1 (seed 42) | sft-v1-seed2 (seed 43) |
|---|---|---|---|
| COBOLEval greedy pass@1 | 0.043 → 0.000 | 0.034 → 0.000 | not measured |
| COBOLEval sampled pass@1 | 0.019 → 0.003, Δ −0.015 [−0.023, −0.009] | 0.021 → 0.021, Δ +0.000 [−0.011, +0.011] | 0.021 → 0.055, Δ +0.035 [+0.017, +0.053] |
| COBOLEval sampled pass@10 | 0.150 → 0.028 | 0.139 → 0.145 | 0.139 → 0.243 |
| CobolCodeBench completion sampled pass@1 | 0.008 → 0.020 | 0.009 → 0.000 | not measured |
| HumanEval-Python sampled pass@1 | 0.969 → 0.934 (Δ −0.034) | 0.969 → 0.919 (Δ −0.050) | not measured |
| Held-out CodeNet tasks | not measured | Δ pass@1 +0.065 [+0.048, +0.083] | Δ pass@1 +0.066 [+0.048, +0.085] |

sft-v1 no longer loses on COBOLEval sampled (seed 42) or gains (seed 43), and it gives the first selection-free evidence that SFT works (held-out tasks). Python forgetting is larger than in sft-v0.

## 8. Limitations

- Two seeds only; they disagree on COBOLEval, so the size of the COBOLEval effect is not known. The guards were measured for seed 42 only.
- Held-out tasks use the CodeNet statement's sample tests (few per task), stdin format only, 5 samples per task.
- The format diagnosis (section 6) was made after the verdict and does not change it.
- One held-out base program (`codenet/p03971`, greedy) exceeded the sandbox memory limit; it counts as `runtime_error` (project-log, "held-out base execution").
- Non-thinking only; thinking mode, CobolCodeBench instruct and slice B are outside the criteria.
