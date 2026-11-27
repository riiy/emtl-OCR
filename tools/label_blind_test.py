#!/usr/bin/env python3
"""Local browser-based blind-test labeler. No OCR engine or external network."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import secrets
import tarfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from blind_test_progress import append_event, counts, read_events, reduce_events

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>emtl-OCR 盲测人工标注</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#20242b;background:#f5f6f8}
body{max-width:760px;margin:32px auto;padding:0 18px}main{background:white;border:1px solid #dfe3e8;border-radius:14px;padding:24px}
h1{font-size:22px;margin:0 0 8px}.muted{color:#68717d;font-size:14px}
#captcha{display:block;width:min(100%,505px);height:auto;image-rendering:pixelated;margin:26px auto;background:#eee}
input{display:block;width:180px;margin:18px auto;padding:12px;font-size:28px;text-align:center;letter-spacing:8px;border:1px solid #aeb7c2;border-radius:8px}
.actions{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}button{padding:10px 18px;border:1px solid #aeb7c2;border-radius:8px;background:#fff;font-size:15px;cursor:pointer}
button.primary{background:#202a36;color:white;border-color:#202a36}button:disabled{opacity:.4;cursor:not-allowed}
nav{display:flex;justify-content:space-between;align-items:center;margin-top:22px}.status{text-align:center;min-height:24px;margin:12px 0}
</style></head><body><main>
<h1>盲测人工标注</h1><div class="muted">仅显示原始图片，不调用或展示任何 OCR 预测。标签必须是四位 ASCII 数字，保留前导零。</div>
<img id="captcha" alt="待人工识别图片"><div id="meta" class="muted" style="text-align:center"></div>
<input id="label" inputmode="numeric" maxlength="4" autocomplete="off" aria-label="四位数字标签">
<div class="actions"><button id="submit" class="primary">提交标签 (Enter)</button><button id="skip">跳过</button></div>
<nav><button id="prev">上一张</button><span id="progress" class="muted"></span><button id="next">下一张</button></nav>
<div id="status" class="status" role="status"></div>
</main><script>
const token=location.hash.slice(1);history.replaceState(null,"",location.pathname);
let state=null,index=0,dirty=false;
const $=id=>document.getElementById(id);
async function api(path,options={}){const r=await fetch(path,{...options,headers:{"Content-Type":"application/json","X-Labeler-Token":token,...options.headers}});const d=await r.json();if(!r.ok)throw Error(d.error||"请求失败");return d}
function current(){return state.items[index]}
function render(){const item=current();if(!item)return;$("captcha").src="/image/"+item.id+"?t="+Date.now();$("meta").textContent=(index+1)+" / "+state.items.length+" · "+item.id;
const existing=state.latest[item.id];$("label").value=existing&&existing.status==="labeled"?existing.label:"";dirty=false;
$("status").textContent=existing?("当前状态："+(existing.status==="labeled"?"已标注":"已跳过")):"未处理";
$("progress").textContent="有效标注 "+state.labeled+" / "+state.target+" · 跳过 "+state.skipped;
$("prev").disabled=index===0;$("next").disabled=index===state.items.length-1;
$("submit").disabled=state.labeled>=state.target&&(!existing||existing.status!=="labeled");
}
async function refresh(){state=await api("/api/state");if(index>=state.items.length)index=state.items.length-1;render()}
async function move(delta){if(dirty&&!confirm("当前输入尚未提交，确定离开并丢弃吗？"))return;index=Math.max(0,Math.min(state.items.length-1,index+delta));render()}
async function act(status){const item=current(),label=$("label").value;if(status==="labeled"&&(!/^[0-9]{4}$/.test(label)||label.length!==4)){alert("请输入恰好四位 ASCII 数字，例如 0042");return}
try{await api("/api/event",{method:"POST",body:JSON.stringify({image_id:item.id,status,label:status==="labeled"?label:null})});dirty=false;await refresh();
if(status==="labeled"&&state.labeled<state.target) {const next=state.items.findIndex((x,i)=>i>index&&!state.latest[x.id]);if(next>=0){index=next;render()}}
else if(status==="skipped"){const next=state.items.findIndex((x,i)=>i>index&&!state.latest[x.id]);if(next>=0){index=next;render()}}
}catch(e){$("status").textContent=e.message}}
$("prev").onclick=()=>move(-1);$("next").onclick=()=>move(1);$("submit").onclick=()=>act("labeled");$("skip").onclick=()=>act("skipped");
$("label").addEventListener("input",()=>{dirty=true;$("label").value=$("label").value.replace(/[^0-9]/g,"").slice(0,4)});
document.addEventListener("keydown",e=>{if(e.key==="Enter"){e.preventDefault();act("labeled")}});
window.addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue=""}});
refresh().catch(e=>$("status").textContent=e.message);
</script></body></html>"""


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Labeler:
    def __init__(self, archive: str, manifest: str, progress: str):
        self.archive_path = Path(archive)
        self.progress = Path(progress)
        if file_sha(self.archive_path) != json.loads(Path(manifest).read_text())["source_archive_sha256"]:
            raise ValueError("Archive SHA-256 mismatch")
        self.manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.items = self.manifest["selected"] + self.manifest["reserve"]
        self.by_id = {item["id"]: item for item in self.items}
        self.archive = tarfile.open(self.archive_path, "r:gz")
        self.members = {m.name: m for m in self.archive.getmembers() if m.isfile()}
        self.events = read_events(self.progress)
        self.latest = reduce_events(self.events, self.manifest)
        self.target = int(self.manifest["target_labels"])
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()

    def state(self):
        with self.lock:
            self.events = read_events(self.progress)
            self.latest = reduce_events(self.events, self.manifest)
            labeled, skipped = counts(self.latest)
            return {"items": self.items, "latest": self.latest, "labeled": labeled,
                    "skipped": skipped, "target": self.target}

    def image(self, image_id: str) -> bytes:
        item = self.by_id.get(image_id)
        if item is None or item["path"] not in self.members:
            raise KeyError("Unknown image")
        data = self.archive.extractfile(self.members[item["path"]]).read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Image SHA-256 mismatch")
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        return data

    def record(self, payload: dict):
        with self.lock:
            item = self.by_id.get(payload.get("image_id"))
            if item is None:
                raise ValueError("Unknown image")
            latest = reduce_events(read_events(self.progress), self.manifest)
            labeled, _ = counts(latest)
            status, label = payload.get("status"), payload.get("label")
            old = latest.get(item["id"])
            if status == "labeled" and labeled >= self.target and not (old and old["status"] == "labeled"):
                raise ValueError(f"盲测有效标签已达到 {self.target} 张")
            return append_event(self.progress, item, status, label)


def make_handler(app: Labeler):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def _authorized(self):
            host = self.headers.get("Host", "")
            if host not in {"127.0.0.1:8765", "localhost:8765"}:
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://127.0.0.1:8765", "http://localhost:8765"}:
                return False
            return secrets.compare_digest(self.headers.get("X-Labeler-Token", ""), app.token)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/" and self.headers.get("Host") in {"127.0.0.1:8765", "localhost:8765"}:
                body = (PAGE.replace('const token=location.hash.slice(1);', 'const token=location.hash.slice(1);')).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
                self.end_headers(); self.wfile.write(body); return
            if not self._authorized():
                self._json(403, {"error": "本地会话校验失败"}); return
            if parsed.path == "/api/state":
                self._json(200, app.state()); return
            if parsed.path.startswith("/image/"):
                try: data = app.image(parsed.path.rsplit("/", 1)[-1])
                except (KeyError, ValueError) as exc: self._json(404, {"error": str(exc)}); return
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
                self.end_headers(); self.wfile.write(data); return
            self._json(404, {"error": "Not found"})

        def do_POST(self):
            if not self._authorized():
                self._json(403, {"error": "本地会话校验失败"}); return
            if self.path != "/api/event":
                self._json(404, {"error": "Not found"}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096: raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                event = app.record(payload)
                self._json(200, {"event": event})
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print("labeler:", fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--progress", default="datasets/private/blind-test-progress.jsonl")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = Labeler(args.archive, args.manifest, args.progress)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://127.0.0.1:{args.port}/#{app.token}"
    print(f"Local blind-test labeler: {url.split('#')[0]}")
    print("Only 127.0.0.1 is exposed. Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping labeler...")
    finally:
        server.server_close()
        app.archive.close()


if __name__ == "__main__":
    main()
