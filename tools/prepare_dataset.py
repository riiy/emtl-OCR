#!/usr/bin/env python3
"""Audit the recovered archive and build deterministic, metadata-only P2 manifests."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

CSV_PATH = "hard_labeling/train_1100.csv"
IMAGE_PREFIX = "images/"
SOURCES = {"ACCEPTED", "REJECTED", "SKIPPED_INVALID_FORMAT"}
LABEL_RE = re.compile(r"^[0-9]{4}$")
TRAIN_SEED = 20261126
BLIND_SEED = 20261127


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_archive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    archive_hash = file_sha256(path)
    with tarfile.open(path, "r:gz") as archive:
        files = {}
        for member in archive.getmembers():
            item_path = PurePosixPath(member.name)
            if item_path.is_absolute() or ".." in item_path.parts or "\\" in member.name:
                raise ValueError("Unsafe archive path: " + member.name)
            if member.isfile():
                if member.name in files:
                    raise ValueError("Duplicate archive member: " + member.name)
                files[member.name] = member
        if CSV_PATH not in files:
            raise ValueError("Missing " + CSV_PATH)
        csv_bytes = archive.extractfile(files[CSV_PATH]).read()
        rows = list(csv.DictReader(csv_bytes.decode("utf-8-sig").splitlines()))
        required = {"filename", "code", "source", "sample_id", "sha256"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError("Training CSV is missing required columns")

        image_names = sorted(
            name for name in files
            if name.startswith(IMAGE_PREFIX) and name.lower().endswith(".img")
        )
        if not image_names:
            raise ValueError("No images found in archive")

        seen_names, seen_ids = set(), set()
        normalized, source_counts = [], Counter()
        leading_zero = 0
        for line, row in enumerate(rows, start=2):
            name = row["filename"].strip()
            code = row["code"].strip()
            source = row["source"].strip()
            sample_id = row["sample_id"].strip()
            expected = row["sha256"].strip().lower()
            safe = PurePosixPath(name)
            if not name.startswith(IMAGE_PREFIX) or safe.is_absolute() or ".." in safe.parts or "\\" in name:
                raise ValueError("Unsafe image path at CSV line " + str(line))
            if name in seen_names or name not in files:
                raise ValueError("Duplicate or missing image reference: " + name)
            if not LABEL_RE.fullmatch(code):
                raise ValueError("Label must be four digits at CSV line " + str(line))
            if source not in SOURCES:
                raise ValueError("Unknown source at CSV line " + str(line))
            if not sample_id or sample_id in seen_ids:
                raise ValueError("Missing or duplicate sample_id at CSV line " + str(line))
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Invalid SHA-256 at CSV line " + str(line))
            data = archive.extractfile(files[name]).read()
            actual = sha256(data)
            if actual != expected:
                raise ValueError("SHA-256 mismatch: " + name)
            if not data.startswith(b"\xff\xd8\xff"):
                raise ValueError("Image is not JPEG encoded: " + name)
            seen_names.add(name)
            seen_ids.add(sample_id)
            source_counts[source] += 1
            leading_zero += int(code.startswith("0"))
            normalized.append({
                "filename": name, "code": code, "source": source,
                "sample_id": sample_id, "sha256": actual,
            })

        candidates = {}
        for name in image_names:
            if name in seen_names:
                continue
            data = archive.extractfile(files[name]).read()
            if not data.startswith(b"\xff\xd8\xff"):
                raise ValueError("Candidate is not JPEG encoded: " + name)
            candidates[name] = sha256(data)

    all_hashes = [row["sha256"] for row in normalized] + list(candidates.values())
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError("Duplicate image content detected by SHA-256")
    if len(normalized) + len(candidates) != len(image_names):
        raise ValueError("Image inventory does not reconcile with CSV references")
    return {
        "archive_sha256": archive_hash,
        "csv_sha256": sha256(csv_bytes),
        "rows": normalized,
        "candidates": candidates,
        "image_count": len(image_names),
        "source_counts": dict(sorted(source_counts.items())),
        "leading_zero_count": leading_zero,
    }


def train_validation_manifest(rows: list[dict[str, str]], archive_hash: str, seed: int) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    rng = random.Random(seed)
    train, validation, by_source = [], [], {}
    for source in sorted(grouped):
        group = sorted(grouped[source], key=lambda row: row["filename"])
        rng.shuffle(group)
        n_train = int(len(group) * 0.8 + 0.5)
        train.extend(group[:n_train])
        validation.extend(group[n_train:])
        by_source[source] = {
            "total": len(group), "train": n_train, "validation": len(group) - n_train
        }
    def public_row(row: dict[str, str]) -> dict[str, str]:
        # Labels stay in the local source CSV; this manifest contains membership only.
        return {key: row[key] for key in ("filename", "source", "sample_id", "sha256")}
    train.sort(key=lambda row: row["filename"])
    validation.sort(key=lambda row: row["filename"])
    return {
        "schema_version": 1,
        "dataset": "captcha_train_1100",
        "archive_sha256": archive_hash,
        "seed": seed,
        "train_fraction": 0.8,
        "counts": {
            "total": len(rows), "train": len(train), "validation": len(validation),
            "by_source": by_source,
        },
        "train": [public_row(row) for row in train],
        "validation": [public_row(row) for row in validation],
    }


def blind_candidates_manifest(candidates: dict[str, str], archive_hash: str, seed: int) -> dict[str, Any]:
    items = sorted(candidates.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    def records(part: list[tuple[str, str]], role: str) -> list[dict[str, str]]:
        return [{"filename": name, "sha256": digest, "role": role} for name, digest in part]
    return {
        "schema_version": 1,
        "dataset": "captcha_blind_test_candidates",
        "archive_sha256": archive_hash,
        "seed": seed,
        "target_labeled_count": 200,
        "candidate_count": len(items),
        "primary": records(items[:200], "primary"),
        "reserve": records(items[200:], "reserve"),
        "annotation_policy": {
            "manual_only": True,
            "ocr_predictions_visible": False,
            "label_pattern": "^[0-9]{4}$",
            "preserve_leading_zero": True,
            "skip_requires_reason": True,
            "replacement_order": "reserve list order",
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".gz":
        with temp.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(payload)
    else:
        temp.write_bytes(payload)
    temp.replace(path)


def prepare(archive: Path, output: Path, gzip_manifests: bool = True) -> dict[str, Any]:
    audit = read_archive(archive)
    split = train_validation_manifest(audit["rows"], audit["archive_sha256"], TRAIN_SEED)
    blind = blind_candidates_manifest(audit["candidates"], audit["archive_sha256"], BLIND_SEED)
    if (len(audit["rows"]), audit["image_count"], len(audit["candidates"])) != (1100, 1387, 287):
        raise SystemExit("Archive inventory differs from approved P2 baseline")
    expected = {
        "total": 1100, "train": 880, "validation": 220,
        "by_source": {
            "ACCEPTED": {"total": 1000, "train": 800, "validation": 200},
            "REJECTED": {"total": 30, "train": 24, "validation": 6},
            "SKIPPED_INVALID_FORMAT": {"total": 70, "train": 56, "validation": 14},
        },
    }
    if split["counts"] != expected:
        raise SystemExit("Stratified split differs from approved 880/220 counts")
    suffix = ".json.gz" if gzip_manifests else ".json"
    write_json(output / ("train-validation-manifest" + suffix), split)
    write_json(output / ("blind-test-candidates" + suffix), blind)
    write_json(output / "dataset-audit.json", {
        "schema_version": 1,
        "archive": archive.name,
        "archive_sha256": audit["archive_sha256"],
        "training_csv_sha256": audit["csv_sha256"],
        "image_count": audit["image_count"],
        "labeled_count": len(audit["rows"]),
        "unreferenced_count": len(audit["candidates"]),
        "source_counts": audit["source_counts"],
        "leading_zero_label_count": audit["leading_zero_count"],
        "all_referenced_image_hashes_verified": True,
        "all_unreferenced_candidate_hashes_computed": True,
        "all_image_sha256_unique": True,
        "labels_in_public_manifests": False,
    })
    return {"audit": audit, "split": split, "blind": blind}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/manifests"))
    parser.add_argument("--no-gzip", action="store_true")
    args = parser.parse_args()
    result = prepare(args.archive, args.output_dir, not args.no_gzip)
    print(json.dumps({
        "image_count": result["audit"]["image_count"],
        "labeled_count": len(result["audit"]["rows"]),
        "candidate_count": len(result["audit"]["candidates"]),
        "split": result["split"]["counts"],
        "blind_primary": len(result["blind"]["primary"]),
        "blind_reserve": len(result["blind"]["reserve"]),
        "train_seed": TRAIN_SEED,
        "blind_seed": BLIND_SEED,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
