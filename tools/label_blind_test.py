#!/usr/bin/env python3
"""Run a localhost-only browser labeler for the OCR blind-test set.

The page displays source images only. This module intentionally imports no OCR
engine and never makes outbound network requests.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import secrets
import tarfile
import threading
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

from blind_test_progress import append_event, latest_states, load_events, load_manifest, summarize

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web_labeler"


class LabelerApplication:
    def __init__(self, archive_path: str | Path, manifest_path: str | Path, progress_path: str | Path):
        self.archive_path = Path(archive_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.progress_path = Path(progress_path).resolve()
        self.manifest, self.allowed = load_manifest(self.manifest_path)
        archive_hash = hashlib.sha256(self.archive_path.read_bytes()).hexdigest()
        if archive_hash != self.manifest.get("source_archive_sha256"):
            raise ValueError("Archive SHA-256 does not match the blind-test manifest")
        self.items = list(self.manifest["selected"]) + list(self.manifest["reserve"])
        self.members: dict[str, tarfile.TarInfo] = {}
        with tarfile.open(self.archive_path, "r:gz") as archive:
            members = {member.name: member for member in archive.getmembers() if member.isfile()}
            for item in self.items:
                member = members.get(item["path"])
                if member is None:
                    raise ValueError(f"Image missing from source archive: {item['path']}")
                self.members[item["id"]] = member
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.events = load_events(self.progress_path, self.allowed)
        self.latest = latest_states(self.events)

    def state(self) -> dict:
        with self.lock:
            summary = summarize(self.events, self.manifest["target_labels"])
            items = []
            for item in self.items:
                event = self.latest.get(item["id"])
                items.append({
                    "id": item["id"],
                    "path": item["path"],
                    "status": event["status"] if event else "unseen",
                    "label": event.get("label") if event else None,
                    "ever_seen": event is not None,
                })
            first_unseen = next((i for i, item in enumerate(items) if not item["ever_seen"]), None)
            first_skipped = next((i for i, item in enumerate(items) if item["status"] == "skipped"), None)
            resume_index = first_unseen if first_unseen is not None else first_skipped
            if resume_index is None:
                resume_index = 0
            return {"summary": summary, "items": items, "resume_index": resume_index}

    def image_bytes(self, image_id: str) -> bytes:
        item = self.allowed.get(image_id)
        if item is None:
            raise KeyError(image_id)
        with tarfile.open(self.archive_path, "r:gz") as archive:
            member = archive.getmember(item["path"])
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read image: {item['path']}")
            data = stream.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Image SHA-256 mismatch: {item['path']}")
        return data

    def record(self, image_id: str, status: str, label: str | None) -> dict:
        with self.lock:
            item = self.allowed.get(image_id)
            if item is None:
                raise ValueError("Unknown image id")
            if status not in {"labeled", "skipped"}:
                raise ValueError("Invalid status")
            if status == "labeled":
                if not isinstance(label, str) or len(label) != 4 or not label.isascii() or not label.isdigit():
                    raise ValueError("Label must be exactly four ASCII digits")
            else:
                label = None
            current = self.latest.get(image_id)
            summary = summarize(self.events, self.manifest["target_labels"])
            if summary["labeled_count"] >= self.manifest["target_labels"]:
                if status == "labeled" and (current is None or current["status"] != "labeled"):
                    raise ValueError("The 200-label target is full; modify an existing labeled image or free a slot first")
                if status == "skipped" and (current is None or current["status"] != "labeled"):
                    raise ValueError("The 200-label target is full; only existing labels can be changed")
            next_seq = max((event.get("seq", index) for index, event in enumerate(self.events, start=1)), default=0) + 1
            event = append_event(self.progress_path, item, status, label, next_seq)
            self.events.append(event)
            self.latest[image_id] = event
            return self.state()


class LabelerHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class LabelerHandler(http.server.BaseHTTPRequestHandler):
    server_version = "emtl-OCR-Labeler/1.0"
    sys_version = ""

    @property
    def app(self) -> LabelerApplication:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        # Avoid logging labels, request bodies, or local filesystem details.
        return

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: dict) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _request_is_local(self, require_origin: bool = False) -> bool:
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            return False
        host = self.headers.get("Host", "")
        allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        if require_origin and origin != f"http://127.0.0.1:{self.server.server_port}":
            return False
        if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
            return False
        fetch_site = self.headers.get("Sec-Fetch-Site")
        if fetch_site and fetch_site not in {"same-origin", "none"}:
            return False
        return True

    def _authorized(self) -> bool:
        return self.headers.get("X-Labeler-Token") == self.app.token

    def do_GET(self) -> None:  # noqa: N802
        if not self._request_is_local():
            self._json(403, {"error": "Local access only"})
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            page = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
            page = page.replace("__LABELER_TOKEN__", self.app.token)
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/app.js":
            self._send(200, (WEB_ROOT / "app.js").read_bytes(), "text/javascript; charset=utf-8")
        elif parsed.path == "/style.css":
            self._send(200, (WEB_ROOT / "style.css").read_bytes(), "text/css; charset=utf-8")
        elif parsed.path == "/api/state":
            if not self._authorized():
                self._json(403, {"error": "Invalid local session"})
                return
            self._json(200, self.app.state())
        elif parsed.path.startswith("/api/image/"):
            if not self._authorized():
                self._json(403, {"error": "Invalid local session"})
                return
            image_id = unquote(parsed.path.rsplit("/", 1)[-1])
            try:
                self._send(200, self.app.image_bytes(image_id), "image/jpeg")
            except KeyError:
                self._json(404, {"error": "Unknown image"})
            except (OSError, ValueError, tarfile.TarError) as exc:
                self._json(422, {"error": str(exc)})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._request_is_local(require_origin=True) or not self._authorized():
            self._json(403, {"error": "Local access denied"})
            return
        if urlparse(self.path).path != "/api/event":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object")
            state = self.app.record(payload.get("image_id"), payload.get("status"), payload.get("label"))
            self._json(200, state)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except OSError as exc:
            self._json(500, {"error": f"Cannot persist progress: {exc}"})


def create_server(app: LabelerApplication, port: int = 8765) -> LabelerHTTPServer:
    server = LabelerHTTPServer(("127.0.0.1", port), LabelerHandler)
    server.app = app  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--progress", default="datasets/private/blind-test-progress.jsonl")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    args = parser.parse_args()

    app = LabelerApplication(args.archive, args.manifest, args.progress)
    server = create_server(app, args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Blind-test labeler listening at {url}")
    print(f"Progress: {app.progress_path}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping labeler; progress is already persisted.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
