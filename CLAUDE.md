# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Status:** skeleton only. `src/onprem_codegen/*` packages are empty; `requirements.txt`, Makefile,
Hydra configs and tests don't exist yet (the train image won't build until `requirements.txt` exists).

## Project goal
Domain-adapt an open code model to a low-resource / legacy programming language, fully self-hostable
(no external model APIs), with every dataset, run, sweep and eval traced in Weights & Biases.
Deliverables: on-prem inference service, reproducible pipeline (data → fine-tune → evaluate → register),
W&B dashboards + model registry, base-vs-adapted evaluation report, live before/after demo.

## Key decisions
- **Target language:** COBOL. Benchmark: COBOLEval, executed with GnuCOBOL.
- **Base model:** `Qwen/Qwen3.8-27B` — dense 27B, hybrid Gated DeltaNet + Gated Attention, multimodal,
  thinking mode by default, Apache 2.0.
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
1. **Collect** — The Stack v1 dedup (COBOL subset; v2 needs a separate Software Heritage download),
   X-COBOL (GitHub), IBM Project CodeNet → `raw-corpus`; CodeNet problems + I/O tests → `task-pool`.
   One manifest per source in `data/manifests/`.
2. **Clean** — exact + MinHash dedup, license filter, **decontaminate against COBOLEval**
   (NeMo Curator or datatrove) → `clean-corpus`
3. **Instruction data via rejection sampling** — the model generates many solutions per `task-pool` task;
   keep only those that compile with GnuCOBOL and pass tests → `sft-dataset`
4. **SFT** — TRL + PEFT (QLoRA) → `adapter`
5. **Preference tuning** — DPO on offline pass/fail pairs (primary). GRPO with compile + test reward
   only as a stretch goal on 2× 80 GB (trainer on one GPU, vLLM on the other, capped generation length).
   → `adapter-rl`
6. **Evaluate** — pass@1 / pass@10 on COBOLEval, compile rate, HumanEval-Python for regression
   (catastrophic forgetting). Per-sample generations logged as W&B Tables.
7. **Register** — W&B Model Registry, aliases `candidate` → `production`.

## W&B conventions
- Every stage consumes its inputs **from W&B artifacts** and logs outputs as new artifact versions,
  so lineage from raw corpus to deployed model is automatic.
- Every run logs: git commit, config, input/output artifacts, Vast instance ID.
- Sweeps are small: learning rate × LoRA rank, few configs, short runs.

## Vast.ai rules
- **Every stage must be restartable on a fresh instance.** Nothing important lives only on instance disk.
- Checkpoint every N steps with resume support; interruptible instances only for resumable jobs.
- Request ≥ 200 GB disk (weights alone ≈ 54 GB bf16). Prefer datacenter / Secure Cloud hosts.
- API keys come from env vars — never baked into images.
- Destroy instances when runs finish.

## Serving
- vLLM (recent version supporting this architecture) with OpenAI-compatible API; SGLang as alternative.
- bf16 + LoRA adapter on 1× 80 GB (live base ↔ adapted toggle for the demo);
  merged FP8 on 48 GB or merged AWQ 4-bit on a 24 GB card for cheaper serving.
- Access via SSH tunnel (`ssh -L 8000:localhost:8000 …`), not a public port.
- Demo client: OpenCode pointed at the local endpoint.

## Repo layout
```
benchmarks/              # eval benchmarks (submodules / pinned clones), never trained on
configs/                 # Hydra configs per stage
data/manifests/          # committed source manifests; rest of data/ is gitignored scratch
docker/                  # train, serve, eval-sandbox images
docs/                    # project brief, setup notes
reports/                 # evaluation report
src/onprem_codegen/      # data/ train/ eval/ serve/ — run as `python -m onprem_codegen.<stage>...`
tests/
docker-compose.yml       # local GPU box only; Vast instances can't run Docker
requirements.txt
```
Entry points (planned): `make data`, `make train`, `make eval`, `make register`.

## Commands (local GPU box)
```bash
cp .env.example .env                                   # WANDB_*, HF_TOKEN, VLLM_VERSION
docker compose run --rm train python -m onprem_codegen.train.sft ...
docker compose up serve                                # 127.0.0.1:8000; models "base" and "cobol" (adapter)
docker compose run --rm eval-sandbox cobc -x prog.cbl  # compile/run generated COBOL in isolation
```
On Vast.ai, see README "Running on Vast.ai" (images pushed to a registry, no compose).

## Data rules
- Each manifest declares `role`: `train`, `tasks_and_train`, `seed` (input to synthetic generation only),
  or `eval_only`. Never train on `eval_only` sources (`benchmarks.yaml`, `mainframe_instruct.yaml`).
- Pin `source.revision` and confirm `license.verified` at download time (currently TODOs).
- Downloads go to `data/raw/` scratch; the real output is a W&B artifact.
- Dataset plan and source priority (start with real COBOL; add Java→COBOL only if data size is the bottleneck):
  [docs/dataset.md](docs/dataset.md).
- Execute generated code only inside the `eval-sandbox` container (no network, read-only, resource-capped).
  Vast instances have no Docker, so execution-based eval runs locally for now.

**Keep README.md in sync:** whenever you add, move or remove a top-level directory or module,
update the "Project structure" section of README.md (and this layout) in the same change.
The same goes for "Key decisions" when a decision above changes.

## Communication style
- Keep explanations simple and answers concise. No extra info beyond what was asked.
