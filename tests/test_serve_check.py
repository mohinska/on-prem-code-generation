"""serve/check.py: both `base` and `cobol` must be listed and answer one request."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from onprem_codegen.serve.check import check


def _server(listed, broken=()):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            self._send(200, {"data": [{"id": m} for m in listed]})

        def do_POST(self):
            model = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"]
            if model in broken:   # vLLM answers 4xx for an adapter it could not load
                return self._send(404, {"error": f"model {model} not found"})
            self._send(200, {"choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"content": f"hello from {model}"}}]})

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


@pytest.fixture
def serve(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)   # VllmClient lets it override the URL
    servers = []

    def start(*a, **kw):
        srv, url = _server(*a, **kw)
        servers.append(srv)
        return url
    yield start
    for s in servers:
        s.shutdown()


def test_both_models_answer(serve):
    assert check(serve(["base", "cobol"])) == {"base": None, "cobol": None}


def test_adapter_missing_from_model_list_is_reported_without_a_request(serve):
    res = check(serve(["base"]))
    assert res["base"] is None and "not listed" in res["cobol"]


def test_adapter_listed_but_failing_is_reported(serve):
    res = check(serve(["base", "cobol"], broken={"cobol"}))
    assert res["base"] is None and "404" in res["cobol"]


def test_unreachable_server_is_reported_for_every_model():
    res = check("http://127.0.0.1:1/v1")    # nothing listens on port 1
    assert set(res) == {"base", "cobol"} and all("unreachable" in e for e in res.values())
