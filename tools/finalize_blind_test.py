#!/usr/bin/env python3
"""Validate the final state of an append-only blind-test event log."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from blind_test_progress import counts, read_events, reduce_events


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize(manifest_path, progress_path, labels_path, freeze_path):
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    events = read_events(progress_path)
    latest = reduce_events(events, manifest)
    labeled_count, skipped_count = counts(latest)
    target = int(manifest["target_labels"])
    if labeled_count != target:
        raise ValueError(f"Expected exactly {target} final labeled images; got {labeled_count}")

    labels = []
    for image_id, event in latest.items():
        if event["status"] != "labeled":
            continue
        labels.append({
            "image_id": image_id,
            "path": event["path"],
            "sha256": event["sha256"],
            "label": event["label"],
        })
    if len({row["image_id"] for row in labels}) != target:
        raise ValueError("Final labels contain duplicate image IDs")
    labels.sort(key=lambda row: row["path"])

    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "path", "sha256", "label"])
        writer.writeheader()
        writer.writerows(labels)

    freeze = {
        "status": "FROZEN",
        "label_count": labeled_count,
        "skipped_count": skipped_count,
        "unique_images_with_events": len(latest),
        "event_count": len(events),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "progress_sha256": sha(progress_path),
        "labels_sha256": sha(labels_path),
    }
    freeze_path = Path(freeze_path)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "progress", "labels", "freeze"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.manifest, args.progress, args.labels, args.freeze), indent=2))


if __name__ == "__main__":
    main()
