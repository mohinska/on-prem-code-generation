"""eval/heldout.py: generate on a (fake) server, execute with a fake executor, paired report."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from onprem_codegen.eval import heldout
from onprem_codegen.prompt import build_prompt
from onprem_codegen.utils import read_jsonl

TASKS = [{"task_id": f"codenet/p{i}", "source": "codenet_cobol", "prompt": f"Task {i}\n\nPrint {i}.",
          "tests": [{"input": "1\n", "output": f"{i}\n"}]} for i in range(4)]
HELDOUT = ["codenet/p0", "codenet/p1", "codenet/p2"]   # p3 is a training task


def _server(log):
    """`cobol` answers every task with a passing program, `base` only task p0 (greedy and sampled alike)."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            log.append(body)
            tid = body["messages"][0]["content"].split("\n")[0].split()[1]
            ok = body["model"] == "cobol" or tid == "0"
            text = f"```cobol\n{'PASS' if ok else 'FAIL'} {tid}\n```"
            choices = [{"index": i, "finish_reason": "stop", "message": {"content": text}} for i in range(body.get("n", 1))]
            out = json.dumps({"choices": choices}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    log = []
    srv = _server(log)
    yield f"http://127.0.0.1:{srv.server_port}/v1", log
    srv.shutdown()


@pytest.fixture
def tasks(tmp_path):
    pool, ids = tmp_path / "pool.jsonl", tmp_path / "ids.json"
    pool.write_text("".join(json.dumps(t) + "\n" for t in TASKS), encoding="utf-8")
    ids.write_text(json.dumps({"task_ids": HELDOUT}), encoding="utf-8")
    return heldout.load_tasks(pool, ids)


def fake_executor(program, tests):
    ok = program.startswith("PASS")
    return {"status": "ok" if ok else "wrong_output", "compiled": True, "passed": ok}


def test_load_tasks_keeps_only_heldout(tasks):
    assert sorted(tasks) == HELDOUT


def test_generate_requests_follow_the_protocol_without_hints(tasks, server, tmp_path):
    url, log = server
    path = heldout.generate(tasks, "cobol", url, tmp_path)
    assert len(log) == 2 * len(HELDOUT)   # greedy x1 and one sampled request (n 5) per task
    by = {(b["messages"][0]["content"], b.get("n", 1)): b for b in log}
    for t in tasks.values():
        prompt = build_prompt(t["prompt"])
        assert by[(prompt, 1)]["temperature"] == 0 and by[(prompt, 5)]["temperature"] > 0   # exactly build_prompt: no hints
        assert by[(prompt, 5)]["chat_template_kwargs"] == {"enable_thinking": False}
        assert by[(prompt, 1)]["model"] == "cobol"
    rows = read_jsonl(path)
    assert len(rows) == len(HELDOUT) * 6
    assert {r["decoding"] for r in rows} == {"greedy", "sampled"}
    assert {r["served_model"] for r in rows} == {"cobol"}


def test_generate_resumes_after_an_interruption(tasks, server, tmp_path):
    url, log = server
    path = heldout.generate(tasks, "base", url, tmp_path)
    full = read_jsonl(path)
    keep = [r for r in full if not (r["task_id"] == "codenet/p2" or r["decoding"] == "sampled" and r["task_id"] == "codenet/p1")]
    path.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    log.clear()
    heldout.generate(tasks, "base", url, tmp_path)
    assert len(log) == 3   # p2 greedy + p2 sampled + p1 sampled
    assert len(read_jsonl(path)) == len(full)
    heldout.generate(tasks, "base", url, tmp_path)
    assert len(log) == 3   # nothing left to do


def test_execute_skips_finished_samples_and_marks_missing_programs(tasks, server, tmp_path):
    url, _ = server
    gens = heldout.generate(tasks, "cobol", url, tmp_path)
    rows = read_jsonl(gens)
    rows[0]["generation"] = "no code here"
    gens.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    calls = []

    def counting(program, tests):
        calls.append(program)
        return fake_executor(program, tests)
    res = heldout.execute_all(tasks, gens, tmp_path, counting)
    out = read_jsonl(res)
    assert len(out) == len(rows) and [r["status"] for r in out if r["status"] == "no_program"] == ["no_program"]
    assert len(calls) == len(HELDOUT)   # identical programs of a task run once
    heldout.execute_all(tasks, gens, tmp_path, counting)
    assert len(calls) == len(HELDOUT) and len(read_jsonl(res)) == len(rows)


def test_report_pairs_the_models_by_task(tasks, server, tmp_path):
    url, _ = server
    results = {}
    for m in ("base", "cobol"):
        d = tmp_path / m
        results[m] = read_jsonl(heldout.execute_all(tasks, heldout.generate(tasks, m, url, d), d, fake_executor))
    rep = heldout.report(results["base"], results["cobol"])
    assert rep["base"]["greedy"]["pass@1"] == pytest.approx(1 / 3)
    assert rep["cobol"]["greedy"]["pass@1"] == 1.0
    assert rep["cobol"]["sampled"]["pass@5"] == 1.0 and rep["base"]["sampled"]["pass@5"] == pytest.approx(1 / 3)
    lo, hi = rep["delta"]["greedy"]["pass@1"]["ci"]
    assert rep["delta"]["greedy"]["pass@1"]["mean"] == pytest.approx(2 / 3)
    assert 0 <= lo <= 2 / 3 <= hi   # 3 tasks: the CI is wide
    assert rep["n_tasks"] == 3
