#!/usr/bin/env python3
"""Audit the public emtl-OCR image, manifest and label files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LABEL_RE = re.compile(r"^[0-9]{4}$")
ALLOWED_SPLITS = {"train", "validation", "test"}
EXPECTED_BASELINE_COUNTS = {
    "train": 880, "validation": 220, "test": 200, "total": 1300
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def membership_sha256(image_ids: set[str]) -> str:
    payload = "\n".join(sorted(image_ids)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Manifest root must be a JSON object")
    return value


def audit_dataset(dataset_dir: Path, verify_images: bool = True,
                  verify_labels: bool = True, verify_splits: bool = True) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    labels_path = dataset_dir / "labels.csv"
    images_dir = dataset_dir / "images"
    if not dataset_dir.is_dir():
        raise ValueError(f"Missing dataset directory: {dataset_dir}")
    allowed_root_entries = {"images", "manifest.json", "labels.csv", "README.md", "LICENSE"}
    actual_root_entries = {path.name for path in dataset_dir.iterdir()}
    unexpected_root_entries = actual_root_entries - allowed_root_entries
    missing_root_entries = {"images", "manifest.json", "labels.csv"} - actual_root_entries
    if unexpected_root_entries or missing_root_entries:
        raise ValueError(
            f"Unexpected dataset root inventory; "
            f"missing={sorted(missing_root_entries)}, "
            f"extra={sorted(unexpected_root_entries)}"
        )
    manifest = read_manifest(manifest_path)

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema_version")
    if manifest.get("dataset_version") != "current":
        raise ValueError("dataset_version must be 'current'")
    if not SHA256_RE.fullmatch(manifest.get("source_archive_sha256", "")):
        raise ValueError("Invalid source_archive_sha256")
    if manifest.get("image_format") != "JPEG":
        raise ValueError("image_format must be JPEG")
    if manifest.get("image_size") != [101, 36]:
        raise ValueError("image_size must be [101, 36]")
    if manifest.get("image_filename") != "<full-sha256>.jpg":
        raise ValueError("Unexpected image_filename policy")

    records = manifest.get("images")
    if not isinstance(records, list):
        raise ValueError("Manifest images must be an array")

    by_id = {}
    split_ids = {name: set() for name in ALLOWED_SPLITS}
    expected_paths = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest image record {index} is not an object")
        if set(record) != {"image_id", "path", "split"}:
            raise ValueError(f"Manifest image record {index} has unexpected fields")
        image_id = record["image_id"]
        path = record["path"]
        split = record["split"]
        if not isinstance(image_id, str) or not SHA256_RE.fullmatch(image_id):
            raise ValueError(f"Invalid image_id at manifest index {index}")
        if image_id in by_id:
            raise ValueError(f"Duplicate image_id: {image_id}")
        if path != f"images/{image_id}.jpg":
            raise ValueError(f"Non-canonical image path: {path!r}")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split for {image_id}: {split!r}")
        by_id[image_id] = record
        split_ids[split].add(image_id)
        expected_paths.add(path)

    if len(by_id) != len(records):
        raise ValueError("Manifest image IDs are not unique")

    actual_files = set()
    if not images_dir.is_dir():
        raise ValueError(f"Missing image directory: {images_dir}")
    for path in images_dir.rglob("*"):
        if path.is_file():
            actual_files.add(path.relative_to(dataset_dir).as_posix())
    if actual_files != expected_paths:
        missing = sorted(expected_paths - actual_files)
        extra = sorted(actual_files - expected_paths)
        raise ValueError(
            f"Image inventory mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    if verify_images:
        for image_id, record in by_id.items():
            image_path = dataset_dir / record["path"]
            if sha256_file(image_path) != image_id:
                raise ValueError(f"SHA-256 mismatch: {record['path']}")
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    if (
                        image.format != "JPEG"
                        or image.size != (101, 36)
                        or image.mode != "RGB"
                    ):
                        raise ValueError(f"Unexpected image properties: {record['path']}")
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                raise ValueError(f"Invalid JPEG image: {record['path']}") from exc

    counts = {name: len(split_ids[name]) for name in ("train", "validation", "test")}
    counts["total"] = len(records)
    if manifest.get("counts") != counts:
        raise ValueError(
            f"Manifest counts do not match image records: "
            f"declared={manifest.get('counts')}, actual={counts}"
        )
    if counts["total"] != sum(counts[name] for name in ALLOWED_SPLITS):
        raise ValueError("Split counts do not sum to total")
    if verify_splits:
        baseline_counts = manifest.get("baseline_counts")
        if baseline_counts != EXPECTED_BASELINE_COUNTS:
            raise ValueError("Baseline split counts do not match the approved dataset")
        baseline = manifest.get("baseline_membership")
        if not isinstance(baseline, dict):
            raise ValueError("Missing baseline_membership")
        if set(baseline) != ALLOWED_SPLITS:
            raise ValueError("baseline_membership must define train, validation and test")
        baseline_sets = {}
        for split_name in ALLOWED_SPLITS:
            members = baseline[split_name]
            if not isinstance(members, list) or any(
                not isinstance(item, str) or not SHA256_RE.fullmatch(item)
                for item in members
            ):
                raise ValueError(f"Invalid baseline members for {split_name}")
            if len(members) != len(set(members)):
                raise ValueError(f"Duplicate baseline member in {split_name}")
            if len(members) != baseline_counts[split_name]:
                raise ValueError(f"Baseline member count mismatch for {split_name}")
            baseline_sets[split_name] = set(members)
        if baseline_sets["train"] & baseline_sets["validation"]:
            raise ValueError("Baseline train and validation memberships overlap")
        if baseline_sets["train"] & baseline_sets["test"]:
            raise ValueError("Baseline train and test memberships overlap")
        if baseline_sets["validation"] & baseline_sets["test"]:
            raise ValueError("Baseline validation and test memberships overlap")
        for split_name in ALLOWED_SPLITS:
            missing_baseline = baseline_sets[split_name] - split_ids[split_name]
            if missing_baseline:
                raise ValueError(
                    f"Baseline {split_name} membership changed or disappeared"
                )
        if split_ids["test"] != baseline_sets["test"]:
            raise ValueError("Public fixed test membership has changed")
        new_ids = set(by_id) - set().union(*baseline_sets.values())
        if any(by_id[image_id]["split"] == "test" for image_id in new_ids):
            raise ValueError("New images may only be added to train or validation")
        expected_test_digest = manifest.get("test_membership_sha256")
        if not SHA256_RE.fullmatch(expected_test_digest or ""):
            raise ValueError("Missing or invalid test_membership_sha256")
        actual_test_digest = membership_sha256(split_ids["test"])
        if actual_test_digest != expected_test_digest:
            raise ValueError("Public fixed test membership has changed")

    if verify_labels:
        with labels_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["image_id", "label"]:
                raise ValueError("labels.csv must contain exactly image_id,label")
            label_rows = list(reader)
        labels = {}
        for row_number, row in enumerate(label_rows, 2):
            image_id = row["image_id"]
            label = row["label"]
            if image_id in labels:
                raise ValueError(f"Duplicate label image_id at CSV line {row_number}")
            if not LABEL_RE.fullmatch(label or ""):
                raise ValueError(f"Invalid four-digit label at CSV line {row_number}")
            labels[image_id] = label
        if set(labels) != set(by_id):
            missing = sorted(set(by_id) - set(labels))
            extra = sorted(set(labels) - set(by_id))
            raise ValueError(
                f"Label/image mismatch; missing={missing[:5]}, extra={extra[:5]}"
            )
        leading_zero_count = sum(label.startswith("0") for label in labels.values())
        label_sha256 = sha256_file(labels_path)
    else:
        label_rows = []
        leading_zero_count = None
        label_sha256 = None

    return {
        "status": "PASS",
        "counts": counts,
        "images_verified": len(records) if verify_images else 0,
        "labels_verified": len(label_rows) if verify_labels else 0,
        "leading_zero_labels": leading_zero_count,
        "labels_sha256": label_sha256,
        "test_membership_sha256": manifest.get("test_membership_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets"))
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--verify-labels", action="store_true")
    parser.add_argument("--verify-splits", action="store_true")
    args = parser.parse_args()
    result = audit_dataset(
        args.dataset,
        verify_images=args.verify_images,
        verify_labels=args.verify_labels,
        verify_splits=args.verify_splits,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
