# Frozen baseline: Qwen/Qwen3.8-27B, no adapter

The adapted model (`sft-v0`) is compared against this table only. Eval configs (`configs/eval/baseline*.yaml`) and the
sampling protocol (`evaluation_protocol` in `data/manifests/benchmarks.yaml`) do not change without a new baseline.
Frozen at commit `a1a88b4`. Pass@1 as fractions; pass@10 only for sampled decoding. Details and runs: `docs/project-log.md`.

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

## 95% CI (noise level)

Bootstrap over tasks (2000 resamples, seed 0) of the per-task pass@k, from the frozen `eval-results` artifacts; `-` where a
task has fewer than k samples. An adapted result counts as better than noise only if its paired Δ CI (`eval.report`) excludes 0.
Reproduce: `python -m onprem_codegen.eval.report --ci-of <results.jsonl>...`

| group | tasks | pass@1 [95% CI] | pass@10 [95% CI] |
|---|---|---|---|
| coboleval/completion/non_thinking/greedy | 146 | 0.043 [0.018, 0.078] | - |
| coboleval/completion/non_thinking/sampled | 146 | 0.019 [0.013, 0.026] | 0.150 [0.105, 0.195] |
| coboleval/completion/thinking/sampled | 146 | 0.051 [0.036, 0.070] | 0.281 [0.212, 0.356] |
| cobolcodebench/completion/non_thinking/greedy | 46 | 0.007 [0.000, 0.022] | - |
| cobolcodebench/completion/non_thinking/sampled | 46 | 0.008 [0.000, 0.018] | 0.048 [0.000, 0.109] |
| cobolcodebench/instruct/non_thinking/greedy | 46 | 0.000 [0.000, 0.000] | - |
| cobolcodebench/instruct/non_thinking/sampled | 46 | 0.001 [0.000, 0.003] | 0.011 [0.000, 0.033] |
| humaneval_python/completion/non_thinking/greedy | 164 | 0.972 [0.949, 0.992] | - |
| humaneval_python/completion/non_thinking/sampled | 164 | 0.969 [0.947, 0.985] | 0.994 [0.982, 1.000] |
| humaneval_python/completion/thinking/sampled | 164 | 0.962 [0.933, 0.985] | 0.976 [0.951, 0.994] |

Limits: no thinking on CobolCodeBench (not run), no greedy in thinking, thinking n 10 instead of 20; COBOLEval thinking is a
lower bound (37.5% of answers hit the 28000-token limit).

## W&B artifacts

Alias names are unique per collection, so each part has its own `baseline-final-*` alias, on both artifacts.

| Part | alias | `eval-generations` | `eval-results` |
|---|---|---|---|
| non-thinking, all three benchmarks | `baseline-final-non-thinking` | v1 | v1 |
| thinking, COBOLEval | `baseline-final-thinking-coboleval` | v2 | v5 |
| thinking, HumanEval-Python | `baseline-final-thinking-humaneval` | v3 | v4 |

The adapted run uses the same configs with `adapter` and `run_name: sft-v0` set.
