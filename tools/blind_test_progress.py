"""Append-only blind-test annotation event log and state reducer."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path


def read_events(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {number}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"Invalid event at line {number}: expected object")
            events.append(event)
    return events


def reduce_events(events: list[dict], manifest: dict) -> dict[str, dict]:
    allowed = {item["id"]: item for item in manifest["selected"] + manifest["reserve"]}
    latest: dict[str, dict] = {}
    for number, event in enumerate(events, 1):
        image_id = event.get("image_id")
        item = allowed.get(image_id)
        if item is None:
            raise ValueError(f"Unknown image_id in event {number}: {image_id!r}")
        if event.get("path") != item["path"] or event.get("sha256") != item["sha256"]:
            raise ValueError(f"Manifest mismatch in event {number}")
        status = event.get("status")
        if status == "labeled":
            label = event.get("label")
            if not isinstance(label, str) or len(label) != 4 or not label.isascii() or not label.isdigit():
                raise ValueError(f"Invalid four-digit ASCII label in event {number}")
        elif status == "skipped":
            if event.get("label") is not None:
                raise ValueError(f"Skipped event {number} must not contain a label")
        else:
            raise ValueError(f"Invalid status in event {number}: {status!r}")
        latest[image_id] = event
    return latest


def append_event(path: str | Path, item: dict, status: str, label: str | None = None) -> dict:
    if status not in {"labeled", "skipped"}:
        raise ValueError("status must be labeled or skipped")
    if status == "labeled":
        if not isinstance(label, str) or len(label) != 4 or not label.isascii() or not label.isdigit():
            raise ValueError("label must contain exactly four ASCII digits")
    elif label is not None:
        raise ValueError("skipped events cannot have a label")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = read_events(path)
    event = {
        "seq": len(events) + 1,
        "image_id": item["id"],
        "path": item["path"],
        "sha256": item["sha256"],
        "status": status,
        "label": label,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def counts(latest: dict[str, dict]) -> tuple[int, int]:
    labeled = sum(event["status"] == "labeled" for event in latest.values())
    skipped = sum(event["status"] == "skipped" for event in latest.values())
    return labeled, skipped
