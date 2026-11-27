import hashlib
import http.client
import json
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "tools"))
from label_blind_test import LabelerApplication, create_server


class WebLabelerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_data = b"not-a-real-jpeg-but-hash-checked"
        self.image_hash = hashlib.sha256(self.image_data).hexdigest()
        self.reserve_data = b"reserve-image-data"
        self.reserve_hash = hashlib.sha256(self.reserve_data).hexdigest()
        self.archive = self.root / "source.tgz"
        image_file = self.root / "sample.img"
        reserve_file = self.root / "reserve.img"
        image_file.write_bytes(self.image_data)
        reserve_file.write_bytes(self.reserve_data)
        with tarfile.open(self.archive, "w:gz") as archive:
            archive.add(image_file, arcname="images/sample.img")
            archive.add(reserve_file, arcname="images/reserve.img")
        archive_hash = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "source_archive_sha256": archive_hash,
            "target_labels": 1,
            "selected": [{"id": "sample", "path": "images/sample.img", "sha256": self.image_hash}],
            "reserve": [{"id": "reserve", "path": "images/reserve.img", "sha256": self.reserve_hash}],
        }))
        self.progress = self.root / "progress.jsonl"
        self.app = LabelerApplication(self.archive, self.manifest, self.progress)
        self.server = create_server(self.app, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method, path, body=None, token=True, origin=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.port)
        headers = {}
        if token:
            headers["X-Labeler-Token"] = self.app.token
        if origin:
            headers["Origin"] = f"http://127.0.0.1:{self.port}"
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = connection.getresponse()
        data = response.read()
        content_type = response.getheader("Content-Type", "")
        connection.close()
        return response.status, data, content_type

    def test_page_and_state_are_served_locally(self):
        status, body, content_type = self.request("GET", "/", token=False, origin=False)
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(self.app.token.encode(), body)
        status, body, _ = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["resume_index"], 0)

    def test_image_endpoint_checks_archive_hash(self):
        status, body, content_type = self.request("GET", "/api/image/sample")
        self.assertEqual(status, 200)
        self.assertEqual(body, self.image_data)
        self.assertEqual(content_type, "image/jpeg")

    def test_submit_preserves_leading_zero_and_persists(self):
        status, body, _ = self.request("POST", "/api/event", {"image_id": "sample", "status": "labeled", "label": "0042"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["summary"]["labeled_count"], 1)
        event = json.loads(self.progress.read_text().strip())
        self.assertEqual(event["label"], "0042")
        self.assertEqual(event["seq"], 1)

    def test_rejects_invalid_label(self):
        status, _, _ = self.request("POST", "/api/event", {"image_id": "sample", "status": "labeled", "label": "42"})
        self.assertEqual(status, 400)
        self.assertFalse(self.progress.exists())

    def test_rejects_cross_origin_and_missing_token(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port)
        connection.request("POST", "/api/event", body=json.dumps({"image_id": "sample", "status": "labeled", "label": "0042"}), headers={
            "Host": f"127.0.0.1:{self.port}",
            "Origin": "http://evil.example",
            "Content-Type": "application/json",
            "X-Labeler-Token": self.app.token,
        })
        response = connection.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        connection.close()

        status, _, _ = self.request("GET", "/api/state", token=False)
        self.assertEqual(status, 403)

    def test_duplicate_revisions_use_last_state(self):
        self.app.record("sample", "labeled", "0001")
        state = self.app.record("sample", "labeled", "0042")
        self.assertEqual(state["summary"]["labeled_count"], 1)
        self.assertEqual(len(self.progress.read_text().splitlines()), 2)

    def test_skipped_item_can_be_labeled_again(self):
        skipped = self.app.record("sample", "skipped", None)
        self.assertEqual(skipped["summary"]["labeled_count"], 0)
        labeled = self.app.record("sample", "labeled", "0007")
        self.assertEqual(labeled["summary"]["labeled_count"], 1)
        self.assertEqual(labeled["items"][0]["label"], "0007")

    def test_capacity_allows_edits_but_not_a_new_label(self):
        self.app.record("sample", "labeled", "0042")
        with self.assertRaisesRegex(ValueError, "target is full"):
            self.app.record("reserve", "labeled", "1234")
        edited = self.app.record("sample", "labeled", "0043")
        self.assertEqual(edited["summary"]["labeled_count"], 1)
        freed = self.app.record("sample", "skipped", None)
        self.assertEqual(freed["summary"]["labeled_count"], 0)
        filled = self.app.record("reserve", "labeled", "1234")
        self.assertEqual(filled["summary"]["labeled_count"], 1)

    def test_resume_skips_processed_skipped_images(self):
        self.app.record("sample", "skipped", None)
        self.assertEqual(self.app.state()["resume_index"], 1)


if __name__ == "__main__":
    unittest.main()
