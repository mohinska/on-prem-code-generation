"""
onprem_codegen/serve/adapter_keys.py — make a text-only LoRA adapter loadable by vLLM.

We train on the text-only class, so the adapter keys are `...model.layers.N...`. vLLM serves Qwen3.8 as the full
multimodal model and looks LoRA weights up as `...model.language_model.layers.N...`; without the prefix it loads the
adapter, matches no module, and `cobol` answers exactly like `base` (no error). Run once after fetching the adapter:

    python -m onprem_codegen.serve.adapter_keys /adapters/cobol
"""

from __future__ import annotations

import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def add_language_model_prefix(adapter_dir: str | Path) -> int:
    """Rename `model.layers.` -> `model.language_model.layers.` in adapter_model.safetensors; returns how many keys changed."""
    path = Path(adapter_dir) / "adapter_model.safetensors"
    tensors = load_file(str(path))
    renamed = {k.replace("model.layers.", "model.language_model.layers.", 1) if "language_model" not in k else k: v
               for k, v in tensors.items()}
    changed = sum(k not in renamed for k in tensors)
    if changed:
        save_file(renamed, str(path), metadata={"format": "pt"})
    return changed


if __name__ == "__main__":
    print(f"renamed {add_language_model_prefix(sys.argv[1])} keys")
