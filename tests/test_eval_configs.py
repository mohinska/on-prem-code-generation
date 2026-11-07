"""configs/eval/sft-v0.yaml must be the frozen baseline protocol, except for what makes it the adapted run."""
from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parent.parent / "configs" / "eval"


def _load(name):
    return yaml.safe_load((CFG / name).read_text(encoding="utf-8"))


def test_sft_v0_differs_from_baseline_only_in_adapter_run_name_model_and_modes():
    base, new = _load("baseline.yaml"), _load("sft-v0.yaml")
    assert new["adapter"] == "sft-v0-adapter:candidate" and new["run_name"] == "sft-v0"
    assert new["server"]["model"] == "cobol" and new["modes"] == ["non_thinking"]
    for cfg in (base, new):
        for key in ("run_name", "adapter", "modes"):
            cfg.pop(key)
        cfg["server"].pop("model")
    assert new == base      # same benchmarks, decodings, seed, concurrency, timeouts, execute settings, ks


def _strip(cfg, *keys):
    for key in ("run_name", "adapter", "modes", *keys):
        cfg.pop(key)
    cfg["server"].pop("model")
    return cfg


def test_sft_v1_differs_from_baseline_only_in_adapter_run_name_model_and_modes():
    base, new = _load("baseline.yaml"), _load("sft-v1.yaml")
    assert new["adapter"] == "sft-v1-adapter:candidate" and new["run_name"] == "sft-v1"
    assert new["server"]["model"] == "cobol" and new["modes"] == ["non_thinking"]
    assert _strip(new) == _strip(base)      # all three benchmarks, same protocol


def test_baseline_v2_is_base_non_thinking_on_prompt_affected_groups_only():
    base, new = _load("baseline.yaml"), _load("baseline-v2.yaml")
    assert new["adapter"] is None and new["run_name"] == "baseline-v2"
    assert new["server"]["model"] == "base" and new["modes"] == ["non_thinking"]
    assert new["benchmarks"] == {"coboleval": ["completion"], "cobolcodebench": ["completion"]}
    assert _strip(new, "benchmarks") == _strip(base, "benchmarks")
