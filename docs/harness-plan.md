# Plan: agentic harness and model container

Goal: run any fine-tuned model privately and drive it with an agent loop
(generate → compile in the sandbox → feed errors back → retry). One command starts the model, one command starts the
agent, and swapping the model means changing one config file.

Status: plan, nothing below is built yet.

## Decisions

| Area | Decision |
|---|---|
| Where the model runs | A GPU server running the serve image. Today that is a Vast.ai instance standing in for an on-prem node; on a real node the same image runs with `docker compose up serve`. |
| Where the agent runs | Any laptop (macOS, Linux, Windows + WSL): Python 3.12 + Docker + ssh. The model is reached through `ssh -L 8000:localhost:8000`. |
| Model interface | An OpenAI-compatible `/v1/chat/completions` endpoint plus a model config. The agent and the client know nothing else about the model. |
| Serving backend | vLLM (already in the stack). llama.cpp for running fully on a laptop is a later, optional backend (new dependency, needs approval, see "Later"). |
| Privacy | Prompts and code never leave the server and the laptop at runtime. See "Privacy checklist". |
| New dependencies | None. |

## Architecture

Four parts, each replaceable without touching the others:

```
 laptop                                              GPU server (Vast now, on-prem later)
 ┌───────────────────────────────┐   ssh tunnel     ┌──────────────────────────────────┐
 │ client: agent CLI | OpenCode  │ ───────────────▶ │ backend: serve image (vLLM)      │
 │ harness: agent loop           │  localhost:8000  │   reads model config:            │
 │ tools:  execute() → sandbox   │                  │   base + adapter | merged weights │
 │         (eval-sandbox image)  │                  └──────────────────────────────────┘
 └───────────────────────────────┘
```

| Part | What it is | How to swap it |
|---|---|---|
| Model config | `configs/models/<name>.yaml`: weights, adapter, served name, chat template switches | add one YAML |
| Backend | a script that turns a model config into a running OpenAI-compatible server | add one `docker/serve-<backend>.sh` |
| Harness | `src/onprem_codegen/agent/`: the loop and its tools | add a tool function |
| Client | agent CLI, OpenCode, or any OpenAI client | point it at `localhost:8000` |

### Model config

One file per model, read by both sides. The backend uses the `serve` block, the harness uses `client`.

```yaml
# configs/models/qwen-sft-v2.yaml
name: cobol                         # served model name, what clients send in "model"
serve:
  base: Qwen/Qwen3.8-27B            # HF id, or a local path on an offline node
  adapter: sft-v2-adapter:candidate # W&B artifact, or null for merged / base-only
  max_model_len: 32768
  max_num_seqs: 256                 # this model's Mamba cache limit on 80 GB
  extra_args: []                    # backend-specific flags, passed through
client:
  enable_thinking: false            # sent as chat_template_kwargs
  sampling: {temperature: 0.2, top_p: 0.95, max_tokens: 4096}
```

Gemma 4, the base model, or a future DPO adapter are each a new YAML. No code change.

### Backend (serve image)

Reuse what exists: `docker/serve.Dockerfile` + `docker/serve-entrypoint.sh`. Changes:

1. The entrypoint reads `MODEL_CONFIG=/configs/models/<name>.yaml` and maps it to the env vars it already uses
   (`MODEL`, `ADAPTER_PATH`, `MAX_MODEL_LEN`, ...). Env vars still win, so `scripts/vast_serve.sh` keeps working.
2. Before `vllm serve`, if `adapter` is a W&B artifact and not on disk yet: `wandb artifact get`, then
   `python -m onprem_codegen.serve.adapter_keys` (the existing key fix). Copy `src/onprem_codegen/serve/` and
   `configs/models/` into the image, and install `wandb` at the version pinned in `requirements.txt`.
3. Privacy env baked in: `HF_HUB_OFFLINE=1` after the first fetch, `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`,
   `--disable-log-requests`.
4. Push the image to a private registry. On Vast, start the instance from that image
   (`vastai create instance ... --image $REG/onprem-codegen-serve:$TAG --env MODEL_CONFIG=...`): no `pip install` at
   start-up, so the server is up as soon as the weights are downloaded.

On a real on-prem node: weights and adapter are copied into the `hf-cache` and `/adapters` volumes once, and the
container runs with no network access.

### Harness (agent loop)

New module `src/onprem_codegen/agent/`, built only from existing pieces:

| Need | Existing code |
|---|---|
| Call the model | `eval.generate.VllmClient` (OpenAI-compatible, already handles thinking traces) |
| Prompt | `prompt.build_messages` (the same contract as SFT and eval) |
| Get the program from an answer | `eval.execute.extract_program` |
| Compile and run, isolated | `eval.execute.execute` → `eval-sandbox` image (no network, read-only, limits) |

The loop, roughly 60 lines:

```
messages = build_messages(task)
for turn in range(max_turns):                         # default 3
    answer  = client.generate(messages, sampling, thinking)[0]
    program = extract_program(answer)
    result  = execute(program, tests)                 # tests=[] → compile only
    if result.passed: return program, turn
    messages += [assistant(answer), user(feedback(result))]   # cobc stderr / failed test, truncated
return best_attempt
```

Tools are plain functions with the signature `(program) -> ExecResult`. Compile and run-tests come first; adding a
tool (for example, run with a given stdin) is one function and one line in the feedback.

Entry point: `python -m onprem_codegen.agent "task text" [--tests file.jsonl] [--model-config ...]`. It prints the
final program and each turn's compile/test status. Logs stay on the laptop (`outputs/agent/`), not in W&B, because
the task text may be private code.

### Measuring it

The loop gives a new number: **pass@1 within N repair turns**. `eval/run.py` gets a `--agent-turns N` option that
calls the same loop on COBOLEval. Benchmark data is public, so these runs log to W&B as usual (base vs adapted,
turns 1 vs 3). Per the project rules a result counts only with a W&B run.

## Privacy checklist

- [ ] Server listens on loopback only; access only through the SSH tunnel, no public port.
- [ ] vLLM request logging off (`--disable-log-requests`), usage stats off.
- [ ] `HF_HUB_OFFLINE=1` and no W&B calls after weights are fetched.
- [ ] Generated code runs only in the sandbox (`--network none`, read-only, limits), never on the host.
- [ ] Agent logs stay local; W&B gets only benchmark runs.
- [ ] OpenCode configured with the local provider only, its sharing and telemetry off.
- [ ] Known limit: on Vast the host provider can see the machine. That is a stand-in for on-prem, stated as such in
      the report. Hosted W&B already sees run metadata (decided in CLAUDE.md).

## Tasks

Each task is one PR, in order. Each ends with a check you can run.

| # | Task | Done when |
|---|---|---|
| 1 | `configs/models/` with `qwen-base`, `qwen-sft-v2`, `gemma4-sft`; entrypoint reads `MODEL_CONFIG` | `docker compose up serve` with each config, `python -m onprem_codegen.serve.check` passes |
| 2 | Serve image fetches the W&B adapter + key fix itself; privacy env; push to registry | Vast instance from the image, no manual setup, `serve.check` passes through the tunnel |
| 3 | `agent/` loop + CLI, unit test with a fake client and the `local` sandbox backend | a COBOLEval task solved end-to-end against the Vast server from a laptop |
| 4 | `eval/run.py --agent-turns N` | W&B runs: base and `sft-v2`, turns 1 and 3, on COBOLEval; numbers in the project log |
| 5 | Laptop quick start in README (Docker, ssh tunnel, agent CLI, OpenCode config) | a teammate starts from a fresh laptop with only the README |

## Later (not in this plan)

- **Fully local laptop backend:** `docker/serve-llamacpp.sh` running `llama-server` on a merged, quantized (GGUF)
  model. Needs ~32 GB RAM on Apple Silicon or ~20 GB VRAM for the 27B model in 4 bits. New dependency (llama.cpp),
  support for the Qwen3.8 hybrid architecture not checked, and the quantized model needs its own eval run.
- Agent as an OpenAI-compatible proxy, so OpenCode gets the repair loop without changes.
- Multi-user access, auth, a GUI.
