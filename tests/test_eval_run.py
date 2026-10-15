"""Eval pipeline: prompts, answer parsing, the vLLM request, metrics and a dry run end to end."""
import json
import math
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from onprem_codegen.eval import generate as g
from onprem_codegen.eval import run as r
from onprem_codegen.eval.benchmarks import Task
from onprem_codegen.eval.metrics import pass_at_k, summarize
from onprem_codegen.prompt import INSTRUCTION, IO_COBOLEVAL, IO_FILES, build_prompt

ROOT = Path(__file__).resolve().parent.parent


def _task(bench="coboleval", lang="cobol", prompt="SKELETON", entry="add"):
    return Task(task_id=f"{bench}/x", benchmark=bench, mode="completion", lang=lang, prompt=prompt,
                entry_point=entry, tests=[], canonical_solution=None)


# ---------- prompts ----------

def test_sft_prompt_is_unchanged():
    assert build_prompt("T") == f"T\n\n{INSTRUCTION}"
    assert "standard input" in INSTRUCTION


def test_benchmark_prompts_use_their_io_sentence():
    assert IO_COBOLEVAL in g.messages_for(_task())[0]["content"]
    assert IO_FILES in g.messages_for(_task(bench="cobolcodebench"))[0]["content"]
    for t in (_task(), _task(bench="cobolcodebench")):
        content = g.messages_for(t)[0]["content"]
        assert "standard input" not in content and content.endswith("Answer with a single ```cobol code block.")


def test_python_prompt():
    [m] = g.messages_for(_task(bench="humaneval_python", lang="python", prompt="def f():\n"))
    assert m["role"] == "user" and "```python\ndef f():\n```" in m["content"]


# ---------- answers ----------

@pytest.mark.parametrize("content,reasoning,answer,trace", [
    ("A", None, "A", ""),
    ("<think>plan</think>\nA", None, "\nA", "<think>plan</think>"),
    ("A", "plan", "A", "plan"),                         # reasoning parser on the server
    ("<think>never closed ```cobol X```", None, "", "<think>never closed ```cobol X```"),
])
def test_split_reasoning(content, reasoning, answer, trace):
    assert g.split_reasoning(content, reasoning) == (answer, trace)


def test_code_inside_thinking_is_not_the_answer():
    answer, _ = g.split_reasoning("<think>```cobol\nDRAFT\n```</think>```cobol\nFINAL\n```", None)
    assert g.program_from_answer(_task(bench="cobolcodebench"), answer) == "FINAL\n"


def test_coboleval_program_is_reordered_like_the_reference():
    src = ("       LINKAGE SECTION.\n       01 LINKED-ITEMS.\n       WORKING-STORAGE SECTION.\n"
           "       01 W PIC 9.\n       PROCEDURE DIVISION.\n           GOBACK.\n")
    out = g.program_from_answer(_task(), f"```cobol\n{src}```")
    lines = [x.strip() for x in out.splitlines()]
    assert lines.index("WORKING-STORAGE SECTION.") < lines.index("LINKAGE SECTION.")
    assert "PROCEDURE DIVISION USING LINKED-ITEMS." in lines


def test_no_code_block_is_no_program():
    assert g.program_from_answer(_task(), "I cannot do that.") is None
    assert g.program_from_answer(_task(bench="humaneval_python", lang="python"), "```cobol\nX\n```") is None


def test_python_body_only_gets_the_prompt_prepended():
    t = _task(bench="humaneval_python", lang="python", prompt="def add(a, b):\n    '''doc'''\n")
    assert g.program_from_answer(t, "```python\ndef add(a, b):\n    return a + b\n```") == \
        "def add(a, b):\n    return a + b\n"
    assert g.program_from_answer(t, "```python\n    return a + b\n```") == t.prompt + "    return a + b\n"


# ---------- vLLM request ----------

class _Server(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Server.seen.append((self.path, body))
        resp = {"choices": [{"index": i, "finish_reason": "length" if i else "stop",
                             "message": {"content": f"ans{i}", "reasoning_content": "why" if i == 0 else None}}
                            for i in reversed(range(body["n"]))],
                "usage": {"completion_tokens": 10 * body["n"]}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_vllm_request_and_response(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    srv = HTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = g.VllmClient(f"http://127.0.0.1:{srv.server_port}/v1", "base", 10)
        out = client.generate([{"role": "user", "content": "hi"}],
                              {"n": 2, "temperature": 0.6, "max_tokens": 64, "seed": 7}, enable_thinking=True)
    finally:
        srv.shutdown()
    path, body = _Server.seen[-1]
    assert path == "/v1/chat/completions"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert (body["model"], body["n"], body["max_tokens"], body["seed"]) == ("base", 2, 64, 7)
    assert [o["content"] for o in out] == ["ans0", "ans1"]          # sorted by index
    assert out[0]["reasoning"] == "why" and out[1]["finish_reason"] == "length"
    assert out[0]["completion_tokens"] == 10


# ---------- sampling plan ----------

def test_sampling_plan_follows_the_manifest():
    proto = r.protocol()
    greedy = r.sampling_plan(proto, "non_thinking", "greedy", 100)
    assert len(greedy) == proto["runs"] and [s["seed"] for s in greedy] == [100, 101, 102]
    assert all(s["n"] == 1 and s["temperature"] == 0.0 for s in greedy)
    [sampled] = r.sampling_plan(proto, "thinking", "sampled", 100)
    assert sampled["max_tokens"] == proto["modes"]["thinking"]["max_new_tokens"] and sampled["n"] >= 10
    assert r.sampling_plan(proto, "thinking", "greedy", 100) is None   # Qwen: no greedy in thinking mode


# ---------- metrics ----------

def test_pass_at_k():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert math.isclose(pass_at_k(10, 3, 1), 0.3)
    assert math.isclose(pass_at_k(4, 1, 2), 1 - (3 * 2) / (4 * 3))   # 1 - C(3,2)/C(4,2)


def _row(task, passed, compiled=True, status=None, finish="stop"):
    return {"task_id": task, "passed": passed, "compiled": compiled, "finish_reason": finish,
            "status": status or ("ok" if passed else "wrong_output")}


def test_summarize():
    rows = [_row("a", True), _row("a", False), _row("b", False, compiled=False, status="no_program", finish="length"),
            _row("b", False)]
    m = summarize(rows, [1, 2, 10])
    assert (m["n_tasks"], m["n_samples"], m["samples_per_task"]) == (2, 4, 2)
    assert math.isclose(m["pass@1"], 0.25) and math.isclose(m["pass@2"], 0.5)
    assert "pass@10" not in m                                     # too few samples
    assert (m["compile_rate"], m["no_program_rate"], m["truncation_rate"]) == (0.75, 0.25, 0.25)
    assert summarize(rows, [1], exclude_tasks={"b"})["pass@1"] == 0.5


def test_nondeterministic_tasks_come_from_the_manifest():
    nd = r.nondeterministic_tasks("cobolcodebench")
    assert "cobolcodebench/task_func_21" in nd and "cobolcodebench/task_func_17" not in nd
    assert r.nondeterministic_tasks("coboleval") == set()


# ---------- end to end ----------

def test_dry_run_end_to_end_and_resume(tmp_path):
    cfg = ROOT / "configs/eval/baseline.yaml"
    local_cfg = tmp_path / "cfg.yaml"
    local_cfg.write_text(cfg.read_text(encoding="utf-8").replace("out_dir: outputs/eval", f"out_dir: {tmp_path.as_posix()}"),
                         encoding="utf-8")
    cmd = [sys.executable, "-m", "onprem_codegen.eval.run", "--config", str(local_cfg), "--dry-run",
           "--limit", "1", "--no-wandb", "--run-name", "t"]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "EXECUTE_MODE": "stub"}
    first = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    out = tmp_path / "t"
    gens = r.read_jsonl(out / "generations.jsonl")
    results = r.read_jsonl(out / "results.jsonl")
    assert len(gens) == len(results) > 0
    assert {x["mode"] for x in gens} == {"thinking", "non_thinking"}
    assert all(x["reasoning"] for x in gens if x["mode"] == "thinking")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "coboleval/completion/non_thinking/greedy" in metrics
    assert "cobolcodebench/instruct/thinking/sampled/deterministic" in metrics

    second = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert "0 to go" in second.stdout
    assert len(r.read_jsonl(out / "results.jsonl")) == len(results)


def test_real_eval_requires_the_real_executor(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "EXECUTE_MODE": "stub"}
    p = subprocess.run([sys.executable, "-m", "onprem_codegen.eval.run", "--no-wandb", "--limit", "1"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "EXECUTE_MODE=real" in p.stderr
