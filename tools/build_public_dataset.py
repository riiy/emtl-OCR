#!/usr/bin/env python3
"""Build the public emtl-OCR dataset from the frozen local source artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

EXPECTED_ARCHIVE_SHA256 = (
    "90cb341f02c7db2658d5a25f3a21da431eb7108b3e56fb24da731ac06cb97494"
)
EXPECTED_SOURCE_CSV_SHA256 = (
    "f1f7c460528c138053fa4aea999e7c3e5bd53d35677ef548cb0ad200662c6d2f"
)
SOURCE_CSV_PATH = "hard_labeling/train_1100.csv"
SEED = 20261127
EXPECTED_COUNTS = {"train": 880, "validation": 220, "test": 200}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LABEL_RE = re.compile(r"^[0-9]{4}$")
BLIND_LABEL_FIELDS = ["image_id", "path", "sha256", "label"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv_bytes(data: bytes, expected_fields: list[str], name: str) -> list[dict]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != expected_fields:
        raise ValueError(
            f"{name} fields must be {expected_fields!r}, got {reader.fieldnames!r}"
        )
    return list(reader)


def verify_jpeg(data: bytes, context: str) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG":
                raise ValueError(f"{context}: expected JPEG, got {image.format}")
            if image.size != (101, 36):
                raise ValueError(f"{context}: expected 101x36, got {image.size}")
            if image.mode != "RGB":
                raise ValueError(f"{context}: expected RGB, got {image.mode}")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"{context}: invalid JPEG image") from exc


def load_frozen_blind_labels(
    labels_path: Path, progress_path: Path, freeze_path: Path
) -> tuple[list[dict], dict]:
    freeze = load_json(freeze_path)
    if freeze.get("status") != "FROZEN":
        raise ValueError("Blind-test freeze status is not FROZEN")
    if freeze.get("label_count") != 200:
        raise ValueError("Blind-test freeze must contain exactly 200 labels")
    if freeze.get("skipped_count") != 5:
        raise ValueError("Unexpected blind-test skipped count")
    if freeze.get("event_count") != 206:
        raise ValueError("Unexpected blind-test event count")

    labels_digest = sha256_file(labels_path)
    if labels_digest != freeze.get("labels_sha256"):
        raise ValueError("Blind-test labels CSV does not match the freeze hash")
    progress_digest = sha256_file(progress_path)
    if progress_digest != freeze.get("progress_sha256"):
        raise ValueError("Blind-test progress JSONL does not match the freeze hash")

    rows = []
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != BLIND_LABEL_FIELDS:
            raise ValueError(
                f"Blind-test CSV fields must be {BLIND_LABEL_FIELDS!r}"
            )
        rows = list(reader)
    if len(rows) != 200:
        raise ValueError(f"Expected 200 blind-test labels, got {len(rows)}")

    by_sha = {}
    for row in rows:
        image_id = row["image_id"]
        path = row["path"]
        digest = row["sha256"]
        label = row["label"]
        if not SHA256_RE.fullmatch(digest):
            raise ValueError(f"Invalid blind-test SHA-256: {digest!r}")
        if image_id != digest[:16]:
            raise ValueError(f"Blind-test short ID does not match SHA-256: {image_id}")
        if path != f"images/{digest}.img":
            raise ValueError(f"Unexpected blind-test source path: {path!r}")
        if not LABEL_RE.fullmatch(label):
            raise ValueError(f"Invalid four-digit blind-test label: {label!r}")
        if digest in by_sha:
            raise ValueError(f"Duplicate blind-test image: {digest}")
        by_sha[digest] = row

    events = []
    with progress_path.open(encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError(f"Progress JSONL has an incomplete line: {line_number}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid progress JSON at line {line_number}") from exc
            events.append(event)

    if len(events) != freeze["event_count"]:
        raise ValueError("Progress event count does not match freeze")
    latest = {}
    for expected_seq, event in enumerate(events, 1):
        if event.get("seq") != expected_seq:
            raise ValueError("Progress sequence is not continuous")
        digest = event.get("sha256")
        image_id = event.get("image_id")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError("Progress event has an invalid SHA-256")
        if image_id != digest[:16]:
            raise ValueError("Progress event image ID does not match SHA-256")
        if event.get("path") != f"images/{digest}.img":
            raise ValueError("Progress event path does not match SHA-256")
        status = event.get("status")
        if status not in {"labeled", "skipped"}:
            raise ValueError(f"Unexpected progress status: {status!r}")
        label = event.get("label")
        if status == "labeled" and (
            not isinstance(label, str) or not LABEL_RE.fullmatch(label)
        ):
            raise ValueError("Progress event has an invalid label")
        if status == "skipped" and label is not None:
            raise ValueError("Skipped progress event unexpectedly has a label")
        latest[digest] = event

    final_counts = Counter(event["status"] for event in latest.values())
    if final_counts != Counter({"labeled": 200, "skipped": 5}):
        raise ValueError(f"Unexpected final progress states: {dict(final_counts)}")
    for digest, event in latest.items():
        if event["status"] == "labeled":
            row = by_sha.get(digest)
            if row is None or row["label"] != event["label"]:
                raise ValueError("Final blind-test CSV differs from progress history")
        elif digest in by_sha:
            raise ValueError("A skipped image appears in final blind-test CSV")

    return rows, freeze


def split_original_rows(rows: list[dict], seed: int = SEED) -> tuple[dict, dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)

    train = {}
    validation = {}
    source_counts = {}
    for source, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: row["filename"])
        random.Random(seed + sum(source.encode("utf-8"))).shuffle(group)
        train_count = round(0.8 * len(group))
        source_counts[source] = {
            "train": train_count,
            "validation": len(group) - train_count,
        }
        for row in group[:train_count]:
            train[row["sha256"]] = row
        for row in group[train_count:]:
            validation[row["sha256"]] = row

    if len(train) != EXPECTED_COUNTS["train"]:
        raise ValueError(f"Expected 880 train images, got {len(train)}")
    if len(validation) != EXPECTED_COUNTS["validation"]:
        raise ValueError(f"Expected 220 validation images, got {len(validation)}")
    if set(train) & set(validation):
        raise ValueError("Train and validation splits overlap")
    return (
        {"train": train, "validation": validation},
        source_counts,
    )


def build_dataset(
    archive_path: Path,
    blind_labels_path: Path,
    blind_progress_path: Path,
    blind_freeze_path: Path,
    output_path: Path,
    seed: int = SEED,
) -> dict:
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite labels or images: {output_path}"
        )
    if seed != SEED:
        raise ValueError(f"This frozen dataset uses seed {SEED}, not {seed}")

    archive_digest = sha256_file(archive_path)
    if archive_digest != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Source archive SHA-256 does not match the audited archive")

    blind_rows, freeze = load_frozen_blind_labels(
        blind_labels_path, blind_progress_path, blind_freeze_path
    )

    temporary_parent = output_path.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.building-", dir=temporary_parent)
    )
    try:
        images_dir = temp_path / "images"
        images_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
                if member.isfile()
            }
            image_members = {
                name: member
                for name, member in members.items()
                if name.startswith("images/") and name.endswith(".img")
            }
            if len(image_members) != 1387:
                raise ValueError(f"Expected 1,387 source images, got {len(image_members)}")

            source_csv_member = members.get(SOURCE_CSV_PATH)
            if source_csv_member is None:
                raise ValueError(f"Missing source label CSV: {SOURCE_CSV_PATH}")
            source_csv_bytes = archive.extractfile(source_csv_member).read()
            if sha256_bytes(source_csv_bytes) != EXPECTED_SOURCE_CSV_SHA256:
                raise ValueError("Source label CSV SHA-256 does not match the audited CSV")
            original_rows = read_csv_bytes(
                source_csv_bytes,
                ["filename", "code", "source", "sample_id", "sha256"],
                "source CSV",
            )
            if len(original_rows) != 1100:
                raise ValueError(f"Expected 1,100 original labels, got {len(original_rows)}")

            original_paths = set()
            original_digests = set()
            for row in original_rows:
                path = row["filename"]
                digest = row["sha256"]
                label = row["code"]
                if path in original_paths:
                    raise ValueError(f"Duplicate original image path: {path}")
                if digest in original_digests:
                    raise ValueError(f"Duplicate original image SHA-256: {digest}")
                if not SHA256_RE.fullmatch(digest):
                    raise ValueError(f"Invalid source image SHA-256: {digest!r}")
                if path != f"images/{digest}.img":
                    raise ValueError(f"Source path does not match SHA-256: {path}")
                if not LABEL_RE.fullmatch(label):
                    raise ValueError(f"Invalid source four-digit label: {label!r}")
                original_paths.add(path)
                original_digests.add(digest)

            split_rows, _ = split_original_rows(original_rows, seed)
            records = []
            labels = {}

            for split_name in ("train", "validation"):
                for digest, row in split_rows[split_name].items():
                    source_path = row["filename"]
                    member = image_members.get(source_path)
                    if member is None:
                        raise ValueError(f"Missing source image: {source_path}")
                    data = archive.extractfile(member).read()
                    if sha256_bytes(data) != digest:
                        raise ValueError(f"Source image hash mismatch: {source_path}")
                    verify_jpeg(data, source_path)
                    output_name = f"{digest}.jpg"
                    (images_dir / output_name).write_bytes(data)
                    records.append({
                        "image_id": digest,
                        "path": f"images/{output_name}",
                        "split": split_name,
                    })
                    labels[digest] = row["code"]

            test_digests = set()
            for row in blind_rows:
                digest = row["sha256"]
                source_path = row["path"]
                if digest in original_digests:
                    raise ValueError("Test image overlaps the original train/validation set")
                if digest in test_digests:
                    raise ValueError(f"Duplicate test image: {digest}")
                member = image_members.get(source_path)
                if member is None:
                    raise ValueError(f"Missing test image: {source_path}")
                data = archive.extractfile(member).read()
                if sha256_bytes(data) != digest:
                    raise ValueError(f"Test image hash mismatch: {source_path}")
                verify_jpeg(data, source_path)
                output_name = f"{digest}.jpg"
                (images_dir / output_name).write_bytes(data)
                records.append({
                    "image_id": digest,
                    "path": f"images/{output_name}",
                    "split": "test",
                })
                labels[digest] = row["label"]
                test_digests.add(digest)

            if len(test_digests) != EXPECTED_COUNTS["test"]:
                raise ValueError(f"Expected 200 test images, got {len(test_digests)}")
            if len(records) != 1300 or len(labels) != 1300:
                raise ValueError("Public dataset must contain exactly 1,300 unique images and labels")

        records.sort(key=lambda item: item["image_id"])
        label_rows = sorted(labels.items())
        split_counts = Counter(item["split"] for item in records)
        counts = {
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "test": split_counts["test"],
            "total": len(records),
        }
        if counts != {"train": 880, "validation": 220, "test": 200, "total": 1300}:
            raise ValueError(f"Unexpected public split counts: {counts}")

        test_membership = "\n".join(sorted(test_digests)).encode("ascii")
        baseline_membership = {
            split_name: sorted(
                item["image_id"] for item in records if item["split"] == split_name
            )
            for split_name in ("train", "validation", "test")
        }
        manifest = {
            "schema_version": 1,
            "dataset_version": "current",
            "name": "emtl-OCR four-digit synthetic CAPTCHA dataset",
            "source_archive_sha256": archive_digest,
            "source_labels_sha256": EXPECTED_SOURCE_CSV_SHA256,
            "seed": seed,
            "split_method": (
                "source-stratified 80/20 for the original 1,100 labeled images; "
                "the 200 manually labeled holdout images form the public fixed test set"
            ),
            "split_policy_note": (
                "Original source categories are intentionally not published. "
                "The explicit image membership in this manifest is authoritative; "
                "the seed and method document provenance and are not sufficient to "
                "reconstruct the hidden source-stratified split."
            ),
            "test_membership_sha256": sha256_bytes(test_membership),
            "baseline_counts": counts,
            "baseline_membership": baseline_membership,
            "image_format": "JPEG",
            "image_size": [101, 36],
            "image_filename": "<full-sha256>.jpg",
            "counts": counts,
            "images": records,
        }
        (temp_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (temp_path / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["image_id", "label"])
            writer.writerows(label_rows)

        summary = {
            "status": "built",
            "counts": counts,
            "source_archive_sha256": archive_digest,
            "source_labels_sha256": EXPECTED_SOURCE_CSV_SHA256,
            "blind_labels_sha256": sha256_file(blind_labels_path),
            "blind_progress_sha256": sha256_file(blind_progress_path),
            "blind_freeze_status": freeze["status"],
            "test_membership_sha256": manifest["test_membership_sha256"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.rename(output_path)
        return summary
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--blind-labels", type=Path, required=True)
    parser.add_argument("--blind-progress", type=Path, required=True)
    parser.add_argument("--blind-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    result = build_dataset(
        args.archive,
        args.blind_labels,
        args.blind_progress,
        args.blind_freeze,
        args.output,
        args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
