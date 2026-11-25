# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Status:** in progress. `data/`, `train/sft.py`, `eval/`, `serve/` (adapter key fix, check) and `prompt.py` exist, with
`requirements.txt`, Makefile, configs and tests. `sft-v0`, `sft-v1`, `sft-v2` done; `sft-v2` meets the COBOLEval criteria
but not the HumanEval-Python guard ([docs/sft-v2.md](docs/sft-v2.md)). Gemma 4 31B comparison (same SFT) done: verdict
success, but not the CobolCodeBench guard ([docs/sft-gemma4.md](docs/sft-gemma4.md)).
Not written yet: DPO/GRPO, W&B registry step (`make register`).

## Project goal
Domain-adapt an open code model to a low-resource / legacy programming language, fully self-hostable
(no external model APIs), with every dataset, run, sweep and eval traced in Weights & Biases.
Deliverables: on-prem inference service, reproducible pipeline (data → fine-tune → evaluate → register),
W&B dashboards + model registry, base-vs-adapted evaluation report, live before/after demo.

## Key decisions
- **Target language:** COBOL. Benchmark: COBOLEval, executed with GnuCOBOL.
- **Base model:** `Qwen/Qwen3.8-27B` — dense 27B, hybrid Gated DeltaNet + Gated Attention, multimodal,
  thinking mode by default, Apache 2.0.
- **Comparison model:** `google/gemma-4-31B-it` — dense 31B, multimodal, Apache 2.0. Same SFT data, settings,
  eval protocol and criteria as sft-v2; config `configs/train/sft_gemma4.yaml`.
- **Compute:** Vast.ai team account. 1× 80 GB by default; scale to 2× 80 GB only for GRPO / final runs.
  Instances are ephemeral.
- **Stack:** TRL + PEFT for training (SFT, DPO, GRPO); Hydra configs per stage;
  dependencies pinned in `requirements.txt` (pip).
- **Tracking:** hosted W&B (known limitation: metadata leaves premises; data is public).
- **Reference paper:** COBOL-Coder (Dau et al., arXiv:2604.03986, https://arxiv.org/html/2604.03986v1) —
  compiler-guided validation + multi-stage similarity filtering for COBOL training data. Our research builds on it.
- First step: baseline eval of the unmodified model on COBOLEval (thinking and non-thinking) to confirm headroom.

## Model-specific notes
- Fine-tune **text only**; freeze / drop the vision encoder.
- Default method: **QLoRA (4-bit) on 1× 80 GB**. bf16 LoRA only on 2× 80 GB.
- LoRA target modules: attention projections + Gated DeltaNet projections + MLP.
  Verify module names from the model code.
- Requires recent `transformers`, `flash-linear-attention`, `causal-conv1d`; versions pinned in `requirements.txt`.
- SFT data uses **non-thinking mode**. Evaluate in both thinking and non-thinking modes.

## Pipeline
1. **Collect** — The Stack v1, full not `-dedup`, we dedup ourselves (COBOL subset; v2 needs a separate Software Heritage download),
   X-COBOL (GitHub), IBM Project CodeNet → `raw-corpus`; CodeNet problems + I/O tests → `task-pool`.
   One manifest per source in `data/manifests/`.
2. **Clean** — exact + MinHash dedup (`datasketch`), **decontaminate against COBOLEval** (n-gram containment),
   → `clean-corpus`. Licenses are only tagged (`license_ok`), by decision not used as a filter.
3. **Instruction data via rejection sampling** — the model generates many solutions per `task-pool` task;
   keep only those that compile with GnuCOBOL and pass tests → `sft-dataset`
4. **SFT** — TRL + PEFT (QLoRA) → `adapter`
5. **Preference tuning** — DPO on offline pass/fail pairs (primary). GRPO with compile + test reward
   only as a stretch goal on 2× 80 GB (trainer on one GPU, vLLM on the other, capped generation length).
   → `adapter-rl`
6. **Evaluate** — pass@1 / pass@10 on COBOLEval, compile rate, HumanEval-Python for regression
   (catastrophic forgetting). Per-sample generations logged as W&B Tables.
7. **Register** — W&B Model Registry, aliases `candidate` → `production`.

## Repo layout
```
benchmarks/              # eval benchmarks, downloaded at pinned commits (gitignored), never trained on
configs/                 # Hydra configs per stage
data/manifests/          # committed source manifests; rest of data/ is gitignored scratch
docker/                  # train, serve, eval-sandbox images
docs/                    # project brief, setup notes, project-log.md (running log), evaluation reports
scripts/                 # Vast.ai bootstrap, serving, SFT queue (sft-v1, sft-v2)
src/onprem_codegen/      # data/ train/ eval/ serve/ — run as `python -m onprem_codegen.<stage>...`
                         # prompt.py: build_prompt, shared by SFT data and eval
tests/
.claude/rules/           # rules split by topic, see below
docker-compose.yml       # local GPU box only; Vast instances can't run Docker
Makefile                 # pipeline entry points
requirements.txt
ruff.toml                # lint config (`make lint`), no auto-formatting
```
Entry points: `make data`, `make sft-data`, `make decontam` (alias `decontaminated` on sft-dataset), `make train`, `make benchmarks` (log benchmark-* artifacts), `make eval` (+ `eval-generate`, `eval-execute`), `make test`, `make lint`. `make register` is planned.

## Commands (local GPU box)
```bash
cp .env.example .env                                   # WANDB_*, HF_TOKEN, VLLM_VERSION
docker compose run --rm train python -m onprem_codegen.train.sft ...
docker compose up serve                                # 127.0.0.1:8000; models "base" and "cobol" (adapter)
docker compose run --rm eval-sandbox cobc -x prog.cbl  # compile/run generated COBOL in isolation
```

## Rules
Split by topic in `.claude/rules/`; files with `paths:` load only when working on matching files.
- [workflow.md](.claude/rules/workflow.md) — communication, stack, seeds, commits, project log, README sync (always)
- [wandb.md](.claude/rules/wandb.md) — W&B conventions (always)
- [data.md](.claude/rules/data.md) — data sources, no leakage, fixed comparison protocol, sandbox execution
- [vast-serving.md](.claude/rules/vast-serving.md) — Vast.ai rules and serving
- [code-style.md](.claude/rules/code-style.md) — code style, lint
- [writing.md](.claude/rules/writing.md) — project log and report writing style
