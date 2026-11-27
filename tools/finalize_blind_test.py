#!/usr/bin/env python3
"""Validate the final state of an append-only manual blind-test log."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from blind_test_progress import latest_states, load_events, load_manifest, sha256_file, summarize


def finalize(manifest_path, progress_path, labels_path, freeze_path):
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest, allowed = load_manifest(manifest_path)
    events = load_events(progress_path, allowed)
    latest = latest_states(events)
    labels = []
    skipped = []
    for image_id, event in latest.items():
        item = allowed[image_id]
        if event["status"] == "labeled":
            labels.append({
                "image_id": image_id,
                "path": item["path"],
                "sha256": item["sha256"],
                "label": event["label"],
            })
        else:
            skipped.append(image_id)

    summary = summarize(events, manifest["target_labels"])
    if summary["labeled_count"] != manifest["target_labels"]:
        raise ValueError(
            f"Expected {manifest['target_labels']} unique effective labels; got {summary['labeled_count']}"
        )
    if len({row["image_id"] for row in labels}) != len(labels):
        raise ValueError("Final labels contain duplicate image ids")

    labels.sort(key=lambda row: row["path"])
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "path", "sha256", "label"])
        writer.writeheader()
        writer.writerows(labels)

    result = {
        "status": "FROZEN",
        "label_count": len(labels),
        "skipped_count": len(skipped),
        "event_count": len(events),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "progress_sha256": sha256_file(progress_path),
        "labels_sha256": sha256_file(labels_path),
    }
    freeze_path = Path(freeze_path)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "progress", "labels", "freeze"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.manifest, args.progress, args.labels, args.freeze), indent=2))


if __name__ == "__main__":
    main()
