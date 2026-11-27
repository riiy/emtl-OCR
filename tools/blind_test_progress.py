#!/usr/bin/env python3
"""Append-only progress log and state reduction for blind-test labeling."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def load_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    items = list(manifest.get("selected", [])) + list(manifest.get("reserve", []))
    allowed: dict[str, dict[str, Any]] = {}
    for item in items:
        image_id = item.get("id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("Manifest contains an invalid image id")
        if image_id in allowed:
            raise ValueError(f"Duplicate image id in manifest: {image_id}")
        if not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"Manifest item is incomplete: {image_id}")
        allowed[image_id] = item
    target = manifest.get("target_labels")
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        raise ValueError("Manifest target_labels must be a positive integer")
    if len(manifest.get("selected", [])) != target:
        raise ValueError("Selected image count does not match target_labels")
    return manifest, allowed


def validate_event(event: dict[str, Any], allowed: dict[str, dict[str, Any]]) -> None:
    image_id = event.get("image_id")
    if image_id not in allowed:
        raise ValueError(f"Unknown image id in progress log: {image_id!r}")
    item = allowed[image_id]
    if event.get("path") != item["path"] or event.get("sha256") != item["sha256"]:
        raise ValueError(f"Manifest path/hash mismatch for image {image_id}")
    status = event.get("status")
    if status == "labeled":
        label = event.get("label")
        if not isinstance(label, str) or len(label) != 4 or not label.isascii() or not label.isdigit():
            raise ValueError(f"Label for image {image_id} must be exactly four ASCII digits")
    elif status == "skipped":
        if event.get("label") is not None:
            raise ValueError(f"Skipped image {image_id} must not contain a label")
    else:
        raise ValueError(f"Unknown event status for image {image_id}: {status!r}")
    seq = event.get("seq")
    if seq is not None and (not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0):
        raise ValueError("Event seq must be a positive integer")


def load_events(path: str | Path, allowed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    progress = Path(path)
    if not progress.exists():
        return []
    raw = progress.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Progress JSONL ends with an incomplete line; preserve it and repair explicitly")
    events: list[dict[str, Any]] = []
    previous_seq = 0
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank line in progress JSONL at line {line_number}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at progress line {line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"Progress line {line_number} must contain a JSON object")
        validate_event(event, allowed)
        seq = event.get("seq")
        if seq is not None:
            if seq <= previous_seq:
                raise ValueError("Progress event sequence numbers must be strictly increasing")
            previous_seq = seq
        events.append(event)
    return events


def latest_states(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        latest[event["image_id"]] = event
    return latest


def summarize(events: list[dict[str, Any]], target: int) -> dict[str, int | bool]:
    latest = latest_states(events)
    labeled = sum(event["status"] == "labeled" for event in latest.values())
    skipped = sum(event["status"] == "skipped" for event in latest.values())
    return {
        "labeled_count": labeled,
        "skipped_count": skipped,
        "processed_count": len(latest),
        "event_count": len(events),
        "target_labels": target,
        "complete": labeled == target,
        "at_capacity": labeled >= target,
    }


def append_event(
    path: str | Path,
    item: dict[str, Any],
    status: str,
    label: str | None,
    seq: int,
) -> dict[str, Any]:
    event = {
        "seq": seq,
        "image_id": item["id"],
        "path": item["path"],
        "sha256": item["sha256"],
        "status": status,
        "label": label,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    validate_event(event, {item["id"]: item})
    progress = Path(path)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
