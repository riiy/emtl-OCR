import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("freeze", ROOT / "tools/finalize_blind_test.py")
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


class FreezeTests(unittest.TestCase):
    def test_freeze_preserves_leading_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            progress = root / "progress.jsonl"
            manifest.write_text(json.dumps({
                "target_labels": 2,
                "selected": [{"id": "a", "path": "a.img", "sha256": "a"}, {"id": "b", "path": "b.img", "sha256": "b"}],
                "reserve": [],
            }))
            events = [
                {"seq": 1, "image_id": "a", "path": "a.img", "sha256": "a", "status": "labeled", "label": "0042"},
                {"seq": 2, "image_id": "b", "path": "b.img", "sha256": "b", "status": "labeled", "label": "9310"},
            ]
            progress.write_text("".join(json.dumps(event) + "\n" for event in events))
            result = freeze.finalize(manifest, progress, root / "labels.csv", root / "freeze.json")
            with (root / "labels.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["label"], "0042")
            self.assertEqual(result["label_count"], 2)

    def test_freeze_rejects_incomplete_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            progress = root / "progress.jsonl"
            manifest.write_text(json.dumps({
                "target_labels": 1,
                "selected": [{"id": "a", "path": "a.img", "sha256": "a"}],
                "reserve": [],
            }))
            progress.write_text(json.dumps({"image_id": "a", "path": "a.img", "sha256": "a", "status": "skipped", "label": None}) + "\n")
            with self.assertRaises(ValueError):
                freeze.finalize(manifest, progress, root / "labels.csv", root / "freeze.json")

    def test_freeze_uses_last_event_for_each_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            progress = root / "progress.jsonl"
            manifest.write_text(json.dumps({
                "target_labels": 2,
                "selected": [{"id": "a", "path": "a.img", "sha256": "a"}, {"id": "b", "path": "b.img", "sha256": "b"}],
                "reserve": [],
            }))
            events = [
                {"seq": 1, "image_id": "a", "path": "a.img", "sha256": "a", "status": "labeled", "label": "0001"},
                {"seq": 2, "image_id": "a", "path": "a.img", "sha256": "a", "status": "labeled", "label": "0042"},
                {"seq": 3, "image_id": "b", "path": "b.img", "sha256": "b", "status": "labeled", "label": "9310"},
            ]
            progress.write_text("".join(json.dumps(event) + "\n" for event in events))
            result = freeze.finalize(manifest, progress, root / "labels.csv", root / "freeze.json")
            with (root / "labels.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["label"] for row in rows], ["0042", "9310"])
            self.assertEqual(result["event_count"], 3)

    def test_freeze_accepts_skipped_then_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            progress = root / "progress.jsonl"
            manifest.write_text(json.dumps({
                "target_labels": 1,
                "selected": [{"id": "a", "path": "a.img", "sha256": "a"}],
                "reserve": [],
            }))
            events = [
                {"seq": 1, "image_id": "a", "path": "a.img", "sha256": "a", "status": "skipped", "label": None},
                {"seq": 2, "image_id": "a", "path": "a.img", "sha256": "a", "status": "labeled", "label": "0007"},
            ]
            progress.write_text("".join(json.dumps(event) + "\n" for event in events))
            result = freeze.finalize(manifest, progress, root / "labels.csv", root / "freeze.json")
            self.assertEqual(result["label_count"], 1)
            self.assertEqual(result["skipped_count"], 0)

    def test_freeze_rejects_incomplete_jsonl_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            progress = root / "progress.jsonl"
            manifest.write_text(json.dumps({
                "target_labels": 1,
                "selected": [{"id": "a", "path": "a.img", "sha256": "a"}],
                "reserve": [],
            }))
            progress.write_text('{"image_id":"a"}')
            with self.assertRaises(ValueError):
                freeze.finalize(manifest, progress, root / "labels.csv", root / "freeze.json")


if __name__ == "__main__":
    unittest.main()
