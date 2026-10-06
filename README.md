# On-Prem CodeGen

This project adapts an open code model to COBOL and serves it on our own hardware, with no external
model APIs. Every dataset, training run and evaluation is tracked in Weights & Biases.

The pipeline runs **data → fine-tune → evaluate → register**. It ends with a self-hosted inference
service and a report comparing the base model with the adapted one. The project brief is in
[docs/one-pager.pdf](docs/one-pager.pdf).

> **Status:** early setup. The directory skeleton and Docker images are in place. No pipeline code
> exists yet, and `requirements.txt` still has to be written, so the train image won't build until
> it exists.

## Project structure

Keep this section up to date whenever you add, move or remove a top-level directory or module.

```
.
├── benchmarks/              # eval benchmarks as git submodules / pinned clones (never trained on)
├── configs/                 # Hydra configs, one group per pipeline stage
├── data/                    # local scratch, gitignored except manifests/
│   └── manifests/           # one YAML per source: URL, revision, license, checksums, W&B artifact
├── docker/
│   ├── train.Dockerfile     # training + data pipeline (TRL, PEFT, CUDA kernels)
│   ├── serve.Dockerfile     # vLLM OpenAI-compatible server
│   ├── serve-entrypoint.sh  # serves the base model plus the LoRA adapter if it's present
│   └── eval-sandbox.Dockerfile  # GnuCOBOL; runs untrusted generated code
├── docs/                    # project brief, setup notes
├── reports/                 # evaluation report, figures
├── scripts/                 # vast_onstart.sh: bootstrap a fresh Vast.ai instance
├── src/onprem_codegen/
│   ├── data/                # collect, clean, dedup, decontaminate, rejection sampling
│   ├── train/               # SFT, DPO, GRPO
│   ├── eval/                # COBOLEval pass@k, compile rate, HumanEval regression
│   └── serve/               # adapter export and merge, demo client helpers
├── tests/
├── docker-compose.yml       # local / on-prem GPU box (not for Vast.ai, see below)
├── .env.example             # required environment variables
└── CLAUDE.md                # working notes for Claude Code
```

## Key decisions

| Area | Decision |
|---|---|
| Target language | **COBOL**. Benchmark: COBOLEval, executed with GnuCOBOL. |
| Base model | **`Qwen/Qwen3.8-27B`**: dense 27B, hybrid Gated DeltaNet + Gated Attention, Apache 2.0. We fine-tune the text part only and leave the vision encoder frozen. |
| Fine-tuning | QLoRA (4-bit) on 1× 80 GB by default. bf16 LoRA only on 2× 80 GB. SFT data is in non-thinking mode. |
| Preference tuning | DPO on offline pass/fail pairs. GRPO with a compile + test reward is a stretch goal on 2× 80 GB. |
| Reference paper | [COBOL-Coder](https://arxiv.org/html/2604.03986v1) (Dau et al., 2026): compiler-guided validation and multi-stage similarity filtering for COBOL training data. Our research builds on it. |
| Instruction data | Rejection sampling: we keep only generations that compile with GnuCOBOL and pass the tests. |
| Data sources | The Stack v1 dedup (COBOL; v1 ships file contents, v2 needs a Software Heritage download), X-COBOL, IBM Project CodeNet (tasks with I/O tests for rejection sampling). The Stack v2 Java as optional seeds for synthetic Java→COBOL. One manifest per source in `data/manifests/`. |
| Data cleaning | Exact + MinHash dedup, license filter, decontamination against COBOLEval, CobolCodeBench and HumanEval. |
| Evaluation | pass@1 and pass@10 on COBOLEval in thinking and non-thinking modes, compile rate, and HumanEval-Python to catch forgetting. |
| Tracking | Hosted W&B. Every stage reads its inputs from W&B artifacts and logs new ones. The model registry uses `candidate` → `production` aliases. |
| Compute | Vast.ai team account. 1× 80 GB by default, 2× 80 GB only for GRPO and final runs. Instances are ephemeral. |
| Serving | vLLM with an OpenAI-compatible API, reached through an SSH tunnel only. OpenCode is the demo client. |
| Config / deps | Hydra configs per stage; pinned `requirements.txt` (pip). |

Known limitation: W&B is hosted, so run metadata leaves our machines. The datasets are public.

## Running on Vast.ai

Vast.ai instances are already Docker containers, so `docker compose` doesn't work on them. Instead,
you push our images to a registry and start the instance directly from an image.

**Without a custom image** (smoke tests): start from the stock PyTorch image and let
`scripts/vast_onstart.sh` clone the repo, install `requirements.txt` and download the model:

```bash
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel \
  --disk 200 --ssh --direct --onstart scripts/vast_onstart.sh
# on the instance, once /var/log/onstart.log ends with "onstart: done":
python -m onprem_codegen.data.upload_smoke     # once per W&B project
python -m onprem_codegen.train.sft             # configs/train/sft_smoke.yaml
```

Dry-run the same pipeline locally first with a random 4-layer model of the same architecture:
`python -m onprem_codegen.train.sft model.tiny=true run_name=sft-smoke-local train.max_steps=6 train.save_steps=3`.

### 1. Build and push the images (once per dependency change)

Build on a Linux x86 machine if you can. On an Apple Silicon Mac, `--platform linux/amd64` works
but compiling the CUDA kernels under emulation is very slow.

```bash
REG=ghcr.io/<org>          # or docker.io/<user>
TAG=$(git rev-parse --short HEAD)

docker buildx build --platform linux/amd64 -f docker/train.Dockerfile \
  -t $REG/onprem-codegen-train:$TAG --push .
docker buildx build --platform linux/amd64 -f docker/serve.Dockerfile \
  --build-arg VLLM_VERSION=<version> -t $REG/onprem-codegen-serve:$TAG --push .
```

### 2. Set secrets once

In the Vast.ai console, open **Account → Environment Variables** and add `WANDB_API_KEY`,
`WANDB_ENTITY`, `WANDB_PROJECT`, `HF_TOKEN` and `GITHUB_TOKEN` (read-only, for cloning). They're injected into every instance. Don't bake
them into images or pass them on the command line.

### 3. Rent an instance

```bash
pip install vastai
vastai set api-key <VAST_API_KEY>

# 1× 80 GB, ≥200 GB disk, datacenter hosts, cheapest first
vastai search offers 'num_gpus=1 gpu_ram>=80 disk_space>=200 datacenter=true reliability>0.98' -o 'dph'

vastai create instance <OFFER_ID> \
  --image $REG/onprem-codegen-train:$TAG \
  --disk 200 --ssh --direct
# private registry: add  --login '-u <user> -p <token> ghcr.io'

vastai ssh-url <INSTANCE_ID>
```

For GRPO or final runs, use `num_gpus=2` instead.

### 4. Run a stage

```bash
ssh -p <PORT> root@<HOST>
cd /workspace && git clone <repo-url> && cd on-prem-code-generation   # path the image expects
tmux new -s run          # so the run survives SSH disconnects
python -m onprem_codegen.train.sft ...
```

Every stage must be restartable on a fresh instance:
- Inputs come from W&B artifacts and outputs go back to W&B.
- Checkpoints are saved regularly, and runs resume from the latest one.
- Don't leave anything important only on the instance disk.

### 5. Serve the model

Start the instance from the `onprem-codegen-serve` image instead. With `--ssh`, Vast replaces the
image entrypoint, so you start the server yourself:

```bash
# on the instance: fetch the adapter, then start vLLM
pip install wandb
wandb artifact get <entity>/<project>/adapter:production --root /adapters/cobol
tmux new -s serve 'serve-entrypoint.sh'

# on your laptop: tunnel the API (no public port)
ssh -p <PORT> root@<HOST> -N -L 8000:localhost:8000
curl localhost:8000/v1/models     # lists "base" and "cobol"
```

### 6. Destroy the instance when you're done

```bash
vastai destroy instance <INSTANCE_ID>
```

### Eval sandbox caveat

The `eval-sandbox` image uses `--network none`, a read-only filesystem and resource limits to
isolate generated code. Those settings only apply under Docker, via `docker-compose.yml`. On a
Vast instance there is no Docker, so the isolation isn't there. Until we decide how to handle
this, run execution-based eval (compiling and testing generated COBOL) locally, or on a machine
where the sandbox runs under Docker.

## Running locally (Docker Compose)

For a machine with an NVIDIA GPU and the NVIDIA Container Toolkit:

```bash
cp .env.example .env        # fill in keys and VLLM_VERSION
docker compose build
docker compose run --rm train bash
docker compose up serve     # API on 127.0.0.1:8000
docker compose run --rm eval-sandbox cobc -x prog.cbl
```
