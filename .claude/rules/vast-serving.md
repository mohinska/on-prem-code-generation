---
paths:
  - "scripts/**"
  - "docker/**"
  - "docker-compose.yml"
  - "src/onprem_codegen/serve/**"
  - "src/onprem_codegen/train/**"
---

# Vast.ai and serving

## Vast.ai rules
- **Every stage must be restartable on a fresh instance.** Nothing important lives only on instance disk.
- Checkpoint every N steps with resume support; interruptible instances only for resumable jobs.
- Request ≥ 200 GB disk (weights alone ≈ 54 GB bf16). Prefer datacenter / Secure Cloud hosts.
- API keys come from env vars — never baked into images.
- Destroy instances when runs finish (`AUTODESTROY=1` in the queue scripts).

## Serving
- vLLM (recent version supporting this architecture) with OpenAI-compatible API; SGLang as alternative.
- bf16 + LoRA adapter on 1× 80 GB (live base ↔ adapted toggle for the demo);
  merged FP8 on 48 GB or merged AWQ 4-bit on a 24 GB card for cheaper serving.
- Access via SSH tunnel (`ssh -L 8000:localhost:8000 …`), not a public port.
- Demo client: OpenCode pointed at the local endpoint.
- On Vast.ai, see README "Running on Vast.ai" (images pushed to a registry, no compose).
