"""Run the public example through real framework types and loopback HTTP."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def test_vector_example_uses_installed_adapter():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            denied = self.headers.get("Authorization") == "Bearer rejected-fixture-token"
            value = (
                {"error": {"code": "unauthorized", "message": "fixture-private-row-marker"}}
                if denied
                else [{"id": "item-1", "score": 0.9}]
            )
            payload = json.dumps(value).encode()
            self.send_response(401 if denied else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {key: value for key, value in os.environ.items() if not key.startswith("OC_")}
    env.update(
        OC_BASE_URL=f"http://127.0.0.1:{server.server_port}",
        OC_TENANT="fixture",
        OC_BEARER="fixture-token",
        OC_DB_USER="fixture-user",
        OC_DB_KEY="fixture-key",
        OC_TABLE="fixture_vectors",
        OC_QUERY_VECTOR="[1,0]",
    )
    script = Path(__file__).resolve().parents[1] / "examples/vector_search.py"

    def run(*args, **changes):
        return subprocess.run(
            [sys.executable, str(script), *args],
            env={**env, **changes},
            capture_output=True,
            text=True,
            timeout=45,
        )

    try:
        result = run()
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"ids": ["item-1"], "similarities": [0.9]}
        assert len(requests) == 1
        path, headers, body = requests[0]
        assert path == "/v1/tenants/fixture/vector/fixture_vectors/topk"
        assert body == {"query": [1, 0], "dim": 2, "k": 5, "metric": "cosine"}
        normalized = {key.lower(): value for key, value in headers.items()}
        assert normalized["authorization"] == "Bearer fixture-token"
        assert normalized["x-oc-db-user"] == "fixture-user"
        assert normalized["x-oc-db-key"] == "fixture-key"
        assert run("--help").returncode == 0
        assert run(OC_QUERY_VECTOR="[true]").returncode != 0
        assert run(OC_BEARER="").returncode != 0
        assert len(requests) == 1
        denied = run(OC_BEARER="rejected-fixture-token")
        assert denied.returncode != 0
        assert "Query failed" in denied.stderr
        assert "fixture-private-row-marker" not in denied.stdout + denied.stderr
        assert "rejected-fixture-token" not in denied.stdout + denied.stderr
        assert len(requests) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
