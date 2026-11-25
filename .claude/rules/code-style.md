---
paths:
  - "src/**"
  - "tests/**"
  - "scripts/**"
---

# Code style

Lint: `make lint` (ruff, [ruff.toml](../../ruff.toml)): line length ≤ 120, import order, unused imports/variables.
No auto-formatting; do not reformat old code in bulk, fix long lines in the code you touch.

## Rules
1. Every CLI module starts with a docstring: what it does, the exact `python -m onprem_codegen...` command,
   inputs and outputs, whether it is restartable.
2. Tunable values live in `configs/` (Hydra YAML). Code keeps only UPPER_CASE constants (paths, group names) at the top.
3. Data passes as dicts / jsonl rows with named keys. Schemas are in `docs/contracts.md`; a new field means
   updating contracts.md in the same change.
4. Compute returns data, display is separate (`judge` returns a dict, `render` makes markdown).
5. Reuse before writing: `utils.read_jsonl`, `utils.git_commit`, `prompt.build_prompt`, `eval.metrics`.
6. Long steps are restartable: append-only jsonl, skip what is done, resume.
7. Every entry point logs git commit, config and input/output artifacts to W&B; `--no-wandb` for local smoke runs.
8. Fail loudly at boundaries (missing artifact, unreachable server, empty result → exception). No `except: pass`.
9. Type hints on public functions; private `_helpers` may skip them.
10. Small functions with one purpose and a one-line docstring when the name is not enough.
11. Tests: `tests/test_<module>.py`, pytest, no GPU or network. Tests that need real `cobc` stay separate
    (like `test_execute_real.py`). Non-trivial logic gets at least one test.
12. Shell: `set -euo pipefail`, parameters as `${VAR:-default}`, secrets only from env.
13. Comments in English, they explain why, not what.

## Good vs bad

### Named data, not tuple indexing
```python
# BAD
lo, hi = res[3][0], res[3][1]
# GOOD
m = _cmp(base, adapted, COBOL_S)
lo, hi = m["ci"]
```

### Configs are data, not code
```python
# BAD
LR, RANK = 2e-4, 16            # hardcoded in train/sft.py
# GOOD: configs/train/sft_smoke.yaml, overridden per run (`lora.r=32`)
# lora: {r: 16, alpha: 32}
# train: {learning_rate: 2e-4}
```

### Compute returns data, display happens separately
```python
# BAD
def judge(...):
    print(f"| main | {'yes' if ok else 'no'} |")
# GOOD
v = judge(base_rows, adapted_rows)   # dict
md = render(v)                      # markdown
```

### Small functions with a single purpose
```python
# BAD: one 80-line main() that downloads, generates, executes and logs
# GOOD
def run_program(source: str, tests: list[dict], timeout_s: float) -> dict:
    """Compile `source` and run it once per {"input", "output"} test."""
```
