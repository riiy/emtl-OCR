import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("freeze", ROOT / "tools/finalize_blind_test.py")
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)

from blind_test_progress import append_event, counts, read_events, reduce_events


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "target_labels": 2,
            "selected": [
                {"id": "a", "path": "a.img", "sha256": "a"},
                {"id": "b", "path": "b.img", "sha256": "b"},
                {"id": "c", "path": "c.img", "sha256": "c"},
            ],
            "reserve": [],
        }

    def test_revision_uses_last_event_and_preserves_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            append_event(path, self.manifest["selected"][0], "labeled", "0042")
            append_event(path, self.manifest["selected"][0], "labeled", "0043")
            events = read_events(path)
            latest = reduce_events(events, self.manifest)
            self.assertEqual(len(events), 2)
            self.assertEqual(latest["a"]["label"], "0043")
            self.assertEqual(counts(latest), (1, 0))

    def test_skipped_can_be_labeled_later(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            append_event(path, self.manifest["selected"][0], "skipped")
            append_event(path, self.manifest["selected"][0], "labeled", "0007")
            latest = reduce_events(read_events(path), self.manifest)
            self.assertEqual(latest["a"]["status"], "labeled")
            self.assertEqual(latest["a"]["label"], "0007")
            self.assertEqual(counts(latest), (1, 0))

    def test_labeled_can_be_skipped_and_count_decreases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            append_event(path, self.manifest["selected"][0], "labeled", "1234")
            append_event(path, self.manifest["selected"][0], "skipped")
            self.assertEqual(counts(reduce_events(read_events(path), self.manifest)), (0, 1))

    def test_rejects_invalid_label_and_corrupt_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            with self.assertRaises(ValueError):
                append_event(path, self.manifest["selected"][0], "labeled", "１２３４")
            path.write_text("{broken\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_events(path)


class FreezeTests(unittest.TestCase):
    def test_freeze_uses_final_state_and_preserves_leading_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            progress = root / "progress.jsonl"
            manifest.write_text(json.dumps(self_manifest()), encoding="utf-8")
            events = [
                {"image_id": "a", "path": "a.img", "sha256": "a", "status": "labeled", "label": "0042"},
                {"image_id": "a", "path": "a.img", "sha256": "a", "status": "labeled", "label": "0009"},
                {"image_id": "b", "path": "b.img", "sha256": "b", "status": "labeled", "label": "9310"},
            ]
            progress.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            result = freeze.finalize(manifest, progress, root / "labels.csv", root / "freeze.json")
            with (root / "labels.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["label"], "0009")
            self.assertEqual(result["event_count"], 3)
            self.assertEqual(result["label_count"], 2)

    def test_freeze_rejects_incomplete_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            progress = root / "progress.jsonl"
            manifest.write_text(json.dumps(self_manifest()), encoding="utf-8")
            progress.write_text(json.dumps({"image_id": "a", "path": "a.img", "sha256": "a", "status": "skipped", "label": None}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                freeze.finalize(manifest, progress, root / "labels.csv", root / "freeze.json")


def self_manifest():
    return {
        "target_labels": 2,
        "selected": [
            {"id": "a", "path": "a.img", "sha256": "a"},
            {"id": "b", "path": "b.img", "sha256": "b"},
        ],
        "reserve": [],
    }


if __name__ == "__main__":
    unittest.main()
