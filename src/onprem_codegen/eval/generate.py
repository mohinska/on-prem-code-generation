"""
onprem_codegen/eval/generate.py — benchmark prompts, vLLM requests and program assembly.

Used by eval/run.py. Needs only an OpenAI-compatible server (vLLM, see docker/serve-entrypoint.sh),
so it runs on the Vast instance next to the server or on a laptop through the SSH tunnel.

Prompts follow docs/contracts.md §3 (`build_prompt`), with the I/O sentence of each benchmark.
HumanEval is Python, outside the COBOL contract: it gets the usual "complete this function" prompt.
Sampling settings per mode come from data/manifests/benchmarks.yaml (evaluation_protocol.modes).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from onprem_codegen.eval.benchmarks import Task
from onprem_codegen.eval.execute import extract_program
from onprem_codegen.prompt import IO_COBOLEVAL, IO_FILES, build_messages

PYTHON_INSTRUCTION = ("Complete the following Python function. Answer with a single ```python code block "
                      "containing the complete function, including its signature and any imports.")

_PY_BLOCK = re.compile(r"```python[^\n]*\n(.*?)```", re.S | re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S)


# ---------- prompts ----------

def messages_for(task: Task) -> list[dict]:
    if task.lang == "python":
        return [{"role": "user", "content": f"{PYTHON_INSTRUCTION}\n\n```python\n{task.prompt}```"}]
    io = IO_COBOLEVAL if task.benchmark == "coboleval" else IO_FILES
    return build_messages(task.prompt, io)


# ---------- answer -> program ----------

def split_reasoning(content: str, reasoning: str | None) -> tuple[str, str]:
    """(answer, reasoning). vLLM with a reasoning parser returns the trace separately; without one it
    stays in `content` as <think>...</think>. An unclosed <think> means the answer was truncated."""
    if reasoning:
        return content, reasoning
    traces = "".join(m.group(0) for m in _THINK.finditer(content))
    answer = _THINK.sub("", content)
    if "<think>" in answer:   # opened, never closed
        return "", traces + answer
    return answer, traces


def swap_sections(src: str) -> str:
    """COBOLEval's own post-processing (scripts/generate.py): the prompt lists LINKAGE before
    WORKING-STORAGE, so put WORKING-STORAGE first and make PROCEDURE DIVISION take LINKED-ITEMS."""
    working_storage, linkage, procedure, begin = [], [], [], []
    current = begin
    for line in src.split("\n"):
        stripped = line.strip().upper()
        if stripped.startswith("WORKING-STORAGE SECTION."):
            current = working_storage
        elif stripped.startswith("LINKAGE SECTION."):
            current = linkage
        elif stripped.startswith("PROCEDURE DIVISION"):
            current = procedure
            line = "       PROCEDURE DIVISION USING LINKED-ITEMS."
        current.append(line)
    return "\n".join(begin + working_storage + linkage + procedure)


def program_from_answer(task: Task, answer: str) -> str | None:
    """The program to execute, or None when the answer has no code block of the task's language."""
    if task.lang == "python":
        m = _PY_BLOCK.search(answer)
        if not m:
            return None
        code = m.group(1)
        # Some answers repeat only the body; then the prompt supplies the signature and imports.
        return code if re.search(rf"def\s+{re.escape(task.entry_point)}\s*\(", code) else task.prompt + code
    program = extract_program(answer)
    if program is not None and task.benchmark == "coboleval":
        program = swap_sections(program)
    return program


# ---------- server ----------

class VllmClient:
    """OpenAI-compatible chat completions; one request returns all n samples of a task."""

    def __init__(self, base_url: str, model: str, timeout_s: float):
        self.url = os.environ.get("VLLM_BASE_URL", base_url).rstrip("/") + "/chat/completions"
        self.model, self.timeout = model, timeout_s

    def generate(self, messages: list[dict], sampling: dict, enable_thinking: bool) -> list[dict]:
        """sampling: n, temperature, top_p, max_tokens, seed, ... (passed through to vLLM).
        Returns one dict per sample: content, reasoning, finish_reason, completion_tokens (request total / n)."""
        body = {"model": self.model, "messages": messages, **sampling,
                "chat_template_kwargs": {"enable_thinking": enable_thinking}}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.load(r)
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 3 or (isinstance(e, urllib.error.HTTPError) and e.code < 500):
                    raise
                time.sleep(2 ** (attempt + 1))
        choices = sorted(resp["choices"], key=lambda c: c["index"])
        tokens = (resp.get("usage") or {}).get("completion_tokens")
        return [{"content": c["message"].get("content") or "",
                 "reasoning": c["message"].get("reasoning_content") or c["message"].get("reasoning"),
                 "finish_reason": c.get("finish_reason"),
                 "completion_tokens": tokens / len(choices) if tokens else None} for c in choices]
