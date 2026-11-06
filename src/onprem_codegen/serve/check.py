"""
onprem_codegen/serve/check.py — A3 rehearsal: `base` and the LoRA adapter `cobol` are both listed by vLLM and answer one request.

    python -m onprem_codegen.serve.check [--base-url http://localhost:8000/v1]     # through the SSH tunnel

Exit code 1 if either model is missing or fails. A LoRA vLLM could not load usually shows up here as a 4xx.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from onprem_codegen.eval.generate import VllmClient

# A fixed COBOL text scored by both models: the adapter must change its token logprobs.
SCORED = ("       IDENTIFICATION DIVISION.\n       PROGRAM-ID. FACTORIAL.\n       DATA DIVISION.\n"
          "       WORKING-STORAGE SECTION.\n       01 N PIC 9(9).\n       PROCEDURE DIVISION.\n"
          "           ACCEPT N FROM STDIN\n           DISPLAY N\n           STOP RUN.")
PROMPT = "Write a complete GnuCOBOL program in fixed format that displays HELLO. Answer with a single ```cobol code block."


def check(base_url: str, models=("base", "cobol")) -> dict[str, str | None]:
    """{model: None if it answered, else the error}."""
    base_url = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=30) as r:
            listed = {m["id"] for m in json.load(r)["data"]}
    except OSError as e:   # tunnel down, or vLLM still loading (the tunnel answers with a reset)
        return {m: f"server unreachable at {base_url}: {e}" for m in models}
    out: dict[str, str | None] = {}
    for model in models:
        if model not in listed:
            out[model] = f"not listed by /models (listed: {sorted(listed)})"
            continue
        try:
            answer = VllmClient(base_url, model, 300).generate(
                [{"role": "user", "content": PROMPT}], {"max_tokens": 128, "temperature": 0}, False)[0]
            out[model] = None if answer["content"].strip() else "empty answer"
        except urllib.error.HTTPError as e:
            out[model] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
    return out


def adapter_effect(base_url: str, adapter: str = "cobol", base: str = "base") -> float:
    """Max |logprob difference| of the adapter vs the base model on SCORED. 0.0 means vLLM loaded the adapter but
    applied nothing (this happened when the adapter keys lacked `language_model`, see serve/adapter_keys.py)."""
    def logprobs(model):
        body = {"model": model, "prompt": SCORED, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 0}
        req = urllib.request.Request(f"{base_url.rstrip('/')}/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return [x for x in json.load(r)["choices"][0]["logprobs"]["token_logprobs"] if x is not None]
    return max(abs(a - b) for a, b in zip(logprobs(base), logprobs(adapter)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Check that base and the cobol adapter both answer.")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    base_url = ap.parse_args().base_url
    res = check(base_url)
    for model, err in res.items():
        print(f"{model:6} {'ok' if err is None else 'FAIL ' + err}")
    failed = any(res.values())
    if not failed:
        effect = adapter_effect(base_url)
        print(f"adapter effect (max logprob diff) {effect:.4f}", "ok" if effect > 1e-3 else "FAIL: adapter is not applied")
        failed = effect <= 1e-3
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
