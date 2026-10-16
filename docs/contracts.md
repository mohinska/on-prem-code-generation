# Team contracts

Interfaces between the three work streams (A: eval + serving, B: real-code corpus, C: tasks + SFT).
Change them only with all three agreeing; bump the artifact version when a schema changes.

## 1. Record schemas (JSONL, one object per line)

All text is UTF-8. `content_sha256` is the SHA-256 of the exact `content` / `completion` string.
Source-specific provenance fields (`provenance_fields_required` in each manifest) go in `meta`,
named exactly as in the manifest.

### `raw-corpus` / `clean-corpus` (B; C adds CodeNet submissions)

```json
{
  "id": "<content_sha256>",
  "source": "stack_v1_cobol | xcobol | codenet_cobol",
  "content": "<full COBOL file>",
  "content_sha256": "…",
  "license": "<SPDX id or null>",
  "meta": {"original_path": "…", "original_license": "…"},
  "syntax_check": {"status": "ok | compile_error | timeout | skipped", "stderr": "<first 500 chars>"}
}
```

`meta` per source:

| source | meta fields |
|---|---|
| `stack_v1_cobol` | `original_path`, `original_license` |
| `xcobol` | `repo_name`, `original_path`, `repo_license` |
| `codenet_cobol` | `problem_id`, `submission_id` |

`.cpy` files (copybooks, not programs) stay in `clean-corpus`; SFT builders skip rows with `ext == ".cpy"`.

`syntax_check` is `execute(content, [])`'s `status` plus truncated `stderr`, filled in the clean stage; `raw-corpus` has `{"status": "skipped", "stderr": ""}`.

### `task-pool` (C)

```json
{
  "task_id": "codenet/p00001",
  "source": "codenet_cobol",
  "prompt": "<plain-text problem statement>",
  "tests": [{"input": "…", "output": "…"}],
  "meta": {"problem_id": "p00001"}
}
```

### `sft-dataset` (C)

Same `prompt` / `completion` shape as `tests/fixtures/sft_smoke.jsonl`, which `train/sft.py` already reads.

```json
{
  "prompt": "<user message, see section 3>",
  "completion": "```cobol\n…\n```",
  "source": "rejection_sampling | codenet_cobol",
  "task_id": "codenet/p00001",
  "content_sha256": "<sha256 of completion>",
  "meta": {"generator_model": "Qwen/Qwen3.8-27B", "submission_id": null}
}
```

`prompt` is the raw user message, **not** the chat-templated string; the trainer applies the template.

### Eval generations (A, logged as a W&B Table)

Columns: `task_id`, `benchmark`, `mode` (`thinking` / `non_thinking`), `sample_idx`, `prompt`,
`generation`, `program`, `status`, `stderr`, `duration_s`.

## 2. Executor API (A owns, B and C call)

Module: `src/onprem_codegen/eval/execute.py`.

```python
def extract_program(text: str) -> str | None:
    """First ```cobol fenced block in a model answer; None if there is none."""

def execute(code: str, tests: list[dict] | str, lang: str = "cobol", timeout_s: float = 10.0) -> ExecResult:
    """Compile `code` with GnuCOBOL, run it once per {"input", "expected"} test (stdin -> stdout).

    ExecResult.status: compile_error = cobc failed; timeout = compile or a run exceeded timeout_s;
    runtime_error = non-zero exit; wrong_output = ran, output differs; ok = compiled and all tests passed.
    The first failing step decides the status. tests=[] compiles only (ok = it compiled).
    Other test shapes (COBOLEval caller, CobolCodeBench files, Python) are listed in the module docstring.
    """
```

Rules:
- Output match: compare stdout to `expected` after stripping trailing whitespace on each line and trailing blank lines.
- `tests=[]` means compile only (`cobc -fsyntax-only`, no run), so `compile_error` is distinct from `timeout`. B uses this for the syntax check.
- Real runs go through the `eval-sandbox` container (no network, read-only, resource caps).
  `cobc` version and flags are fixed once in `data/manifests/benchmarks.yaml` and read from there.
- COBOLEval has its own test harness. A wraps it behind the same `ExecResult` type, so callers never see the difference.

## 3. Prompt format and chat template

One user message, no system prompt. Built in one place, `build_prompt(task_text)`, used by both the
SFT data and eval. Generating SFT data and running eval with different prompts would make the base
vs adapted comparison meaningless.

```text
{task_text}

Write a complete GnuCOBOL program in fixed format. Read input from standard input and write the
result to standard output. Answer with a single ```cobol code block.
```

- **I/O sentence:** the second sentence depends on how the task does I/O; everything else is fixed.
  `build_prompt(task_text, io=IO_STDIN)` is the default for task-pool and SFT data. Benchmarks whose
  programs do not use stdin/stdout pass their own sentence (`prompt.py`): `IO_COBOLEVAL` (complete the
  given program, keep PROGRAM-ID and LINKAGE SECTION, store the answer in RESULT) and `IO_FILES`
  (CobolCodeBench: read and write the named files). HumanEval is Python and outside this contract.
- **Template:** the base model tokenizer's own chat template, via `apply_chat_template`. No custom template.
- **SFT:** `enable_thinking=False`, as in `train/sft.py` (`to_prompt_completion`).
- **Eval:** both modes. Through vLLM's OpenAI API, pass `chat_template_kwargs={"enable_thinking": false}`
  for non-thinking; this renders the same string as the trainer.
- **Answer:** one ```` ```cobol ```` block, fixed format (as in the smoke fixture). `extract_program`
  takes the first block; anything outside it is ignored.
- **Thinking traces** are never put in `completion`.
- Sampling settings for each mode live in `data/manifests/benchmarks.yaml`, not here.
