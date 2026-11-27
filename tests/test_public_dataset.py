import csv
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_dataset
import build_public_dataset


def make_jpeg(path: Path, color=(255, 255, 255)) -> str:
    image = Image.new("RGB", (101, 36), color)
    image.save(path, format="JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SplitTests(unittest.TestCase):
    def test_source_stratified_split_is_deterministic_and_preserves_counts(self):
        rows = []
        for source, count in (
            ("group_a", 1000),
            ("group_b", 30),
            ("group_c", 70),
        ):
            for index in range(count):
                digest = f"{len(rows) + 1:064x}"
                rows.append({
                    "filename": f"images/{digest}.img",
                    "code": f"{index % 10000:04d}",
                    "source": source,
                    "sample_id": str(index),
                    "sha256": digest,
                })
        first, source_counts = build_public_dataset.split_original_rows(rows)
        second, _ = build_public_dataset.split_original_rows(rows)
        self.assertEqual(set(first["train"]), set(second["train"]))
        self.assertEqual(set(first["validation"]), set(second["validation"]))
        self.assertEqual(len(first["train"]), 880)
        self.assertEqual(len(first["validation"]), 220)
        self.assertEqual(sorted(
            (item["train"], item["validation"])
            for item in source_counts.values()
        ), [(24, 6), (56, 14), (800, 200)])

    def test_builder_refuses_to_overwrite_existing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "datasets"
            output.mkdir()
            labels_path = output / "labels.csv"
            labels_path.write_text("keep\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_public_dataset.build_dataset(
                    root / "missing.tgz",
                    root / "missing-labels.csv",
                    root / "missing-progress.jsonl",
                    root / "missing-freeze.json",
                    output,
                )
            self.assertEqual(labels_path.read_text(encoding="utf-8"), "keep\n")


class PublicDatasetAuditTests(unittest.TestCase):
    def build_small_dataset(self, root: Path):
        images = root / "images"
        images.mkdir(parents=True)
        records = []
        labels = []
        split_values = ("train", "validation", "test")
        for index, split in enumerate(split_values, 1):
            path = images / f"{index:064x}.jpg"
            digest = make_jpeg(path, color=(index * 20, index * 20, index * 20))
            path.rename(images / f"{digest}.jpg")
            records.append({
                "image_id": digest,
                "path": f"images/{digest}.jpg",
                "split": split,
            })
            labels.append((digest, f"{index - 1:04d}"))
        baseline = {
            split: sorted(item["image_id"] for item in records if item["split"] == split)
            for split in split_values
        }
        test_digest = hashlib.sha256(
            "\n".join(baseline["test"]).encode("ascii")
        ).hexdigest()
        manifest = {
            "schema_version": 1,
            "dataset_version": "current",
            "source_archive_sha256": "a" * 64,
            "source_labels_sha256": "b" * 64,
            "seed": 20261127,
            "split_method": "test",
            "split_policy_note": "test",
            "test_membership_sha256": test_digest,
            "baseline_counts": {"train": 1, "validation": 1, "test": 1, "total": 3},
            "baseline_membership": baseline,
            "image_format": "JPEG",
            "image_size": [101, 36],
            "image_filename": "<full-sha256>.jpg",
            "counts": {"train": 1, "validation": 1, "test": 1, "total": 3},
            "images": sorted(records, key=lambda item: item["image_id"]),
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with (root / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["image_id", "label"])
            writer.writerows(sorted(labels))
        (root / "README.md").write_text("test\n", encoding="utf-8")

    def test_audit_accepts_valid_dataset_and_preserves_leading_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_small_dataset(root)
            with patch.object(audit_dataset, "EXPECTED_BASELINE_COUNTS", {
                "train": 1, "validation": 1, "test": 1, "total": 3
            }):
                result = audit_dataset.audit_dataset(root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["counts"]["total"], 3)
            self.assertEqual(result["leading_zero_labels"], 3)

    def test_audit_rejects_modified_image_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_small_dataset(root)
            image_path = next((root / "images").glob("*.jpg"))
            image_path.write_bytes(image_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                audit_dataset.audit_dataset(root)

    def test_audit_rejects_missing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_small_dataset(root)
            labels_path = root / "labels.csv"
            with labels_path.open(encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            labels_path.write_text(
                "\n".join(",".join(row) for row in rows[:-1]) + "\n",
                encoding="utf-8",
            )
            with patch.object(audit_dataset, "EXPECTED_BASELINE_COUNTS", {
                "train": 1, "validation": 1, "test": 1, "total": 3
            }):
                with self.assertRaisesRegex(ValueError, "Label/image mismatch"):
                    audit_dataset.audit_dataset(root)

    def test_audit_rejects_test_membership_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_small_dataset(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            test_record = next(item for item in manifest["images"] if item["split"] == "test")
            test_record["split"] = "train"
            manifest["counts"] = {"train": 2, "validation": 1, "test": 0, "total": 3}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_dataset.audit_dataset(root)

    def test_audit_rejects_new_test_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_small_dataset(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            images = root / "images"
            digest = make_jpeg(images / "temp.jpg", color=(10, 20, 30))
            (images / "temp.jpg").rename(images / f"{digest}.jpg")
            manifest["images"].append({
                "image_id": digest,
                "path": f"images/{digest}.jpg",
                "split": "test",
            })
            manifest["counts"] = {"train": 1, "validation": 1, "test": 2, "total": 4}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(audit_dataset, "EXPECTED_BASELINE_COUNTS", {
                "train": 1, "validation": 1, "test": 1, "total": 3
            }):
                with self.assertRaisesRegex(ValueError, "test membership has changed"):
                    audit_dataset.audit_dataset(root)

    def test_audit_allows_new_train_or_validation_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_small_dataset(root)
            images = root / "images"
            new_path = images / "new.jpg"
            digest = make_jpeg(new_path, color=(40, 50, 60))
            new_path.rename(images / f"{digest}.jpg")

            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["images"].append({
                "image_id": digest,
                "path": f"images/{digest}.jpg",
                "split": "train",
            })
            manifest["images"].sort(key=lambda item: item["image_id"])
            manifest["counts"] = {
                "train": 2, "validation": 1, "test": 1, "total": 4
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            labels_path = root / "labels.csv"
            with labels_path.open("a", encoding="utf-8", newline="") as stream:
                csv.writer(stream, lineterminator="\n").writerow([digest, "0042"])

            with patch.object(audit_dataset, "EXPECTED_BASELINE_COUNTS", {
                "train": 1, "validation": 1, "test": 1, "total": 3
            }):
                result = audit_dataset.audit_dataset(root)
            self.assertEqual(result["counts"]["total"], 4)

    def test_audit_rejects_unlisted_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_small_dataset(root)
            make_jpeg(root / "images" / "extra.jpg")
            with self.assertRaisesRegex(ValueError, "Image inventory mismatch"):
                audit_dataset.audit_dataset(root)


if __name__ == "__main__":
    unittest.main()
