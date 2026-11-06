"""serve/adapter_keys.py: the text-only adapter has `model.layers.N`, vLLM wants `model.language_model.layers.N`."""
import torch
from safetensors.torch import load_file, save_file

from onprem_codegen.serve.adapter_keys import add_language_model_prefix

OLD = "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight"
NEW = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight"
DONE = "base_model.model.model.language_model.layers.1.self_attn.q_proj.lora_B.weight"


def _adapter(tmp_path, keys):
    save_file({k: torch.full((2, 2), float(i)) for i, k in enumerate(keys)}, str(tmp_path / "adapter_model.safetensors"))
    return tmp_path


def test_text_only_keys_get_the_language_model_prefix(tmp_path):
    d = _adapter(tmp_path, [OLD, DONE])
    assert add_language_model_prefix(d) == 1
    t = load_file(str(d / "adapter_model.safetensors"))
    assert set(t) == {NEW, DONE} and t[NEW][0, 0] == 0.0 and t[DONE][0, 0] == 1.0     # values follow their keys


def test_running_it_again_changes_nothing(tmp_path):
    d = _adapter(tmp_path, [OLD])
    add_language_model_prefix(d)
    assert add_language_model_prefix(d) == 0
    assert set(load_file(str(d / "adapter_model.safetensors"))) == {NEW}
