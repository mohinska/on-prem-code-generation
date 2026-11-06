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

PROMPT = "Write a complete GnuCOBOL program in fixed format that displays HELLO. Answer with a single ```cobol code block."


def check(base_url: str, models=("base", "cobol")) -> dict[str, str | None]:
    """{model: None if it answered, else the error}."""
    base_url = base_url.rstrip("/")
    with urllib.request.urlopen(f"{base_url}/models", timeout=30) as r:
        listed = {m["id"] for m in json.load(r)["data"]}
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Check that base and the cobol adapter both answer.")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    res = check(ap.parse_args().base_url)
    for model, err in res.items():
        print(f"{model:6} {'ok' if err is None else 'FAIL ' + err}")
    raise SystemExit(any(res.values()))


if __name__ == "__main__":
    main()
