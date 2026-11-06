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
