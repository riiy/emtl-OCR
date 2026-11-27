import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from label_blind_test import make_handler


class WebLabelerHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = type("FakeApp", (), {})()
        self.app.port = 0
        self.app.token = "test-session-token"
        self.app.items = []
        self.app.manifest = {"selected": [], "reserve": []}
        self.app.progress = Path(self.temp.name) / "progress.jsonl"
        self.app.target = 200
        self.app.lock = threading.RLock()
        self.app.latest = {}
        self.app.events = []
        self.app.state = lambda: {"items": [], "latest": {}, "labeled": 0, "skipped": 0, "target": 200}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        self.app.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.conn = http.client.HTTPConnection("127.0.0.1", self.app.port)

    def tearDown(self):
        self.conn.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_root_sets_local_session_cookie_and_serves_static_page(self):
        self.conn.request("GET", "/", headers={"Host": f"127.0.0.1:{self.app.port}"})
        response = self.conn.getresponse()
        body = response.read().decode()
        self.assertEqual(response.status, 200)
        self.assertIn("盲测人工标注", body)
        self.assertIn("Set-Cookie", response.headers)
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])

    def test_api_rejects_request_without_session_token(self):
        self.conn.request("GET", "/api/state", headers={"Host": f"127.0.0.1:{self.app.port}"})
        response = self.conn.getresponse()
        response.read()
        self.assertEqual(response.status, 403)

    def test_api_rejects_non_local_origin(self):
        self.conn.request("GET", "/api/state", headers={
            "Host": f"127.0.0.1:{self.app.port}",
            "Origin": "https://example.com",
            "X-Labeler-Token": self.app.token,
        })
        response = self.conn.getresponse()
        response.read()
        self.assertEqual(response.status, 403)


if __name__ == "__main__":
    unittest.main()
