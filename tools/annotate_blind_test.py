#!/usr/bin/env python3
"""Run a localhost-only, resumable manual annotator for the 200-image blind set."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

LABEL_RE = re.compile(r"^[0-9]{4}$")
DRAFT_RE = re.compile(r"^[0-9]{0,3}$")

HTML = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>emtl-OCR 人工盲测标注</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;background:Canvas;color:CanvasText}
.sub,.note{color:GrayText;font-size:.9rem}.stats{display:flex;gap:10px;margin:18px 0}
.stat{flex:1;border:1px solid #8886;border-radius:9px;padding:10px}.stat b{display:block;font-size:1.3rem}
.bar{height:8px;background:#8883;border-radius:8px;overflow:hidden}.bar>div{height:100%;background:#299b86;width:0}
.panel{border:1px solid #8886;border-radius:12px;padding:16px;margin-top:18px}
.meta{display:flex;justify-content:space-between;color:GrayText;font-size:.85rem;gap:10px}
img{display:block;max-width:100%;width:606px;height:auto;margin:20px auto;background:#eee}
form,.actions{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:12px}
input{font:600 1.5rem monospace;letter-spacing:.18em;width:150px;padding:9px;text-align:center}
button{padding:10px 13px;border:1px solid #8888;border-radius:8px;background:ButtonFace;color:ButtonText;cursor:pointer}
button.primary{background:#256b5e;color:white;border-color:#256b5e}button:disabled{opacity:.4;cursor:not-allowed}
.warn{color:#b45309}.frozen{border-color:#299b86;background:#299b8615}
@media(max-width:560px){.stats{gap:5px}.stat{padding:7px}.actions button{flex:1}}
</style></head><body>
<h2>emtl-OCR · 200 张盲测人工标注</h2>
<p class="sub">只显示原始图片，不显示 tiny-CNN、ddddOCR 或任何 OCR 预测结果。</p>
<div class="stats">
<div class="stat"><b id="labels">0 / 200</b><span>有效标签</span></div>
<div class="stat"><b id="skips">0</b><span>跳过</span></div>
<div class="stat"><b id="remaining">287</b><span>未处理候选</span></div>
</div>
<div class="bar"><div id="barfill"></div></div>
<section class="panel" id="panel">
<div class="meta"><span id="position">读取中</span><span id="role"></span></div>
<img id="image" alt="待人工标注图片">
<form id="form"><label for="code">人工标签</label>
<input id="code" type="text" inputmode="numeric" maxlength="4" autocomplete="off" placeholder="四位数字">
<button class="primary" id="save" type="submit">保存并下一张</button></form>
<div class="actions">
<button id="prev" type="button">上一张</button>
<button id="next" type="button">下一张待标注</button>
<button id="skip" type="button">无法辨认，跳过</button>
<button id="freeze" type="button" disabled>冻结 200 张标签</button>
</div>
<p class="note" id="message" role="status">四位数字标签自动保存；输入草稿也会自动保存。前导零必须保留。</p>
</section>
<p class="note">快捷键：Enter 保存并下一张；← 上一张；→ 下一张待标注。关闭浏览器后重新运行命令即可恢复进度。</p>
<script>
var state=null,index=0,timer=null;
function el(id){return document.getElementById(id);}
function message(text,warn){el("message").textContent=text;el("message").className="note"+(warn?" warn":"");}
async function api(path,options){
  var response=await fetch(path,Object.assign({cache:"no-store"},options||{}));
  var body=await response.json();
  if(!response.ok)throw new Error(body.error||("HTTP "+response.status));
  return body;
}
function current(){return state.items[index];}
function counts(){
  var labels=state.items.filter(function(x){return x.status==="labeled";}).length;
  var skips=state.items.filter(function(x){return x.status==="skipped";}).length;
  return {labels:labels,skips:skips,remaining:state.items.length-labels-skips};
}
function render(){
  if(!state)return;
  var c=counts(),item=current();
  el("labels").textContent=c.labels+" / "+state.target;
  el("skips").textContent=String(c.skips);
  el("remaining").textContent=String(c.remaining);
  el("barfill").style.width=Math.min(100,c.labels/state.target*100)+"%";
  el("position").textContent="候选 "+(index+1)+" / "+state.items.length+" · "+item.filename;
  el("role").textContent=item.role==="primary"?"首选样本":"备用样本";
  el("image").src="/api/image/"+encodeURIComponent(item.filename);
  el("code").value=item.code||item.draft||"";
  el("code").disabled=state.frozen||(c.labels>=state.target&&item.status!=="labeled");
  el("save").disabled=el("code").disabled;
  el("skip").disabled=state.frozen||(c.labels>=state.target&&item.status!=="labeled");
  el("prev").disabled=index===0;
  el("next").disabled=state.frozen||c.remaining===0;
  el("freeze").disabled=state.frozen||c.labels<state.target;
  el("freeze").textContent=state.frozen?"已冻结":"冻结 "+state.target+" 张标签";
  el("panel").classList.toggle("frozen",state.frozen);
  if(item.status==="labeled")message("已保存："+item.code+"。冻结前可以回看并修改。");
  else if(item.status==="skipped")message("已跳过。原因："+(item.reason||"未填写"));
  else if(c.labels>=state.target)message("已达到 200 个有效标签。请检查并冻结。");
  else if(c.remaining===0)message("287 张候选已处理完，但有效标签不足 200；备用图片已用尽，不得复制或伪造样本。",true);
  else message("输入四位数字自动保存；无法辨认时必须填写跳过原因。");
}
async function refresh(){state=await api("/api/state");render();}
async function saveRecord(filename,status,code,reason){
  var result=await api("/api/annotation",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filename:filename,status:status,code:code,reason:reason})});
  state=result.state;render();
}
function nextPending(){
  for(var i=index+1;i<state.items.length;i++)if(state.items[i].status==="pending")return i;
  for(var j=0;j<=index;j++)if(state.items[j].status==="pending")return j;
  return index;
}
el("code").addEventListener("input",function(){
  clearTimeout(timer);
  var value=el("code").value,filename=current().filename;
  if(!/^[0-9]{0,4}$/.test(value)){message("只允许输入数字。",true);return;}
  timer=setTimeout(async function(){
    try{
      if(value.length===4){await saveRecord(filename,"labeled",value,"");message("自动保存成功："+value);}
      else {await saveRecord(filename,"draft",value,"");message("输入草稿已自动保存。");}
    }catch(error){message(error.message,true);}
  },450);
});
el("form").addEventListener("submit",async function(event){
  event.preventDefault();clearTimeout(timer);
  var value=el("code").value,filename=current().filename;
  if(!/^[0-9]{4}$/.test(value)){message("标签必须是恰好四位数字。",true);return;}
  try{await saveRecord(filename,"labeled",value,"");if(counts().labels<state.target){index=nextPending();render();}}
  catch(error){message(error.message,true);}
});
el("skip").addEventListener("click",async function(){
  clearTimeout(timer);var reason=prompt("请说明无法辨认的原因（必填）：");
  if(reason===null)return;if(!reason.trim()){message("跳过必须记录原因。",true);return;}
  try{await saveRecord(current().filename,"skipped","",reason.trim());if(counts().labels<state.target){index=nextPending();render();}}
  catch(error){message(error.message,true);}
});
el("prev").addEventListener("click",function(){if(index>0){index--;render();}});
el("next").addEventListener("click",function(){index=nextPending();render();});
el("freeze").addEventListener("click",async function(){
  if(!confirm("确认冻结按候选顺序选出的 200 张人工标签？冻结后不可修改。"))return;
  try{var result=await api("/api/freeze",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});state=result.state;render();message("冻结完成。输出保存在本机："+result.output_dir);}
  catch(error){message(error.message,true);}
});
document.addEventListener("keydown",function(event){
  if(event.target===el("code")){if(event.key==="ArrowLeft"&&index>0){event.preventDefault();index--;render();}return;}
  if(event.key==="ArrowLeft"&&index>0){index--;render();}
  if(event.key==="ArrowRight"&&!state.frozen){index=nextPending();render();}
});
refresh().catch(function(error){message(error.message,true);});
</script></body></html>"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_manifest(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    return value


def candidate_items(manifest: dict[str, Any]) -> list[dict[str, str]]:
    items = []
    seen = set()
    for role, records in (("primary", manifest.get("primary", [])), ("reserve", manifest.get("reserve", []))):
        for record in records:
            filename = record.get("filename", "")
            parts = PurePosixPath(filename).parts
            digest = record.get("sha256", "")
            if len(parts) != 2 or parts[0] != "images" or ".." in parts:
                raise ValueError("Unsafe candidate path: " + filename)
            if filename in seen or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Duplicate candidate or invalid SHA-256: " + filename)
            seen.add(filename)
            items.append({"filename": filename, "sha256": digest, "role": role})
    return items


def extract_candidates(archive_path: Path, items: list[dict[str, str]], image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    expected = {item["filename"]: item["sha256"] for item in items}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for filename, digest in expected.items():
            if filename not in members:
                raise ValueError("Candidate missing from archive: " + filename)
            data = archive.extractfile(members[filename]).read()
            if sha256(data) != digest or not data.startswith(b"\xff\xd8\xff"):
                raise ValueError("Candidate hash/JPEG validation failed: " + filename)
            destination = image_dir / Path(filename).name
            if destination.exists():
                if file_sha256(destination) != digest:
                    raise ValueError("Existing extracted image differs: " + filename)
            else:
                temp = destination.with_suffix(destination.suffix + ".tmp")
                temp.write_bytes(data)
                os.replace(temp, destination)


def count_labeled(state: dict[str, Any]) -> int:
    return sum(1 for item in state["annotations"].values() if item.get("status") == "labeled")


def select_final(items: list[dict[str, str]], annotations: dict[str, Any], target: int) -> list[dict[str, Any]]:
    selected = []
    for index, item in enumerate(items, start=1):
        value = annotations.get(item["filename"], {})
        if value.get("status") == "labeled":
            code = value.get("code", "")
            if not LABEL_RE.fullmatch(code):
                raise ValueError("Invalid frozen label: " + item["filename"])
            selected.append({
                "order": len(selected) + 1,
                "candidate_index": index,
                "filename": item["filename"],
                "code": code,
                "role": item["role"],
                "sha256": item["sha256"],
            })
            if len(selected) == target:
                break
    if len(selected) != target:
        raise ValueError("Need %d valid labels; found %d" % (target, len(selected)))
    return selected


class Application:
    def __init__(self, archive: Path, manifest_path: Path, workdir: Path):
        self.archive = archive.resolve()
        self.manifest_path = manifest_path.resolve()
        self.workdir = workdir.resolve()
        self.manifest = load_manifest(self.manifest_path)
        self.items = candidate_items(self.manifest)
        self.target = int(self.manifest["target_labeled_count"])
        if len(self.items) != int(self.manifest["candidate_count"]):
            raise ValueError("Candidate count mismatch")
        if len(self.manifest["primary"]) != self.target:
            raise ValueError("Primary list must contain target_labeled_count images")
        self.archive_hash = file_sha256(self.archive)
        if self.archive_hash != self.manifest["archive_sha256"]:
            raise ValueError("Archive SHA-256 does not match candidate manifest")
        self.manifest_hash = file_sha256(self.manifest_path)
        self.image_dir = self.workdir / "images"
        self.state_path = self.workdir / "annotations.json"
        self.lock = threading.RLock()
        extract_candidates(self.archive, self.items, self.image_dir)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("archive_sha256") != self.archive_hash or self.state.get("manifest_sha256") != self.manifest_hash:
                raise ValueError("Existing annotation state belongs to another archive/manifest")
        else:
            self.state = {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "archive_sha256": self.archive_hash,
                "manifest_sha256": self.manifest_hash,
                "seed": self.manifest["seed"],
                "target": self.target,
                "frozen": False,
                "annotations": {},
            }
            atomic_json(self.state_path, self.state)

    def snapshot(self) -> dict[str, Any]:
        annotations = self.state["annotations"]
        return {
            "target": self.target,
            "frozen": bool(self.state.get("frozen")),
            "items": [{
                **item,
                "status": annotations.get(item["filename"], {}).get("status", "pending"),
                "code": annotations.get(item["filename"], {}).get("code", ""),
                "draft": annotations.get(item["filename"], {}).get("draft", ""),
                "reason": annotations.get(item["filename"], {}).get("reason", ""),
            } for item in self.items],
        }

    def annotate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.state.get("frozen"):
                raise ValueError("Labels are frozen and cannot be changed")
            filename = payload.get("filename", "")
            if filename not in {item["filename"] for item in self.items}:
                raise ValueError("Unknown candidate filename")
            status = payload.get("status")
            previous = self.state["annotations"].get(filename, {})
            if status == "draft":
                draft = payload.get("code", "")
                if not isinstance(draft, str) or not DRAFT_RE.fullmatch(draft):
                    raise ValueError("Draft must contain zero to three digits")
                if count_labeled(self.state) >= self.target and previous.get("status") != "labeled":
                    raise ValueError("Target of %d valid labels has been reached" % self.target)
                value = {"status": "pending", "code": "", "draft": draft, "reason": ""}
            elif status == "labeled":
                code = payload.get("code", "")
                if not isinstance(code, str) or not LABEL_RE.fullmatch(code):
                    raise ValueError("Label must be exactly four digits; leading zero is preserved")
                if count_labeled(self.state) >= self.target and previous.get("status") != "labeled":
                    raise ValueError("Target of %d valid labels has been reached" % self.target)
                value = {"status": "labeled", "code": code, "draft": "", "reason": ""}
            elif status == "skipped":
                reason = payload.get("reason", "")
                if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 300:
                    raise ValueError("Skip reason is required (maximum 300 characters)")
                value = {"status": "skipped", "code": "", "draft": "", "reason": reason.strip()}
            else:
                raise ValueError("Invalid annotation status")
            value["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.state["annotations"][filename] = value
            atomic_json(self.state_path, self.state)
            return self.snapshot()

    def freeze(self) -> dict[str, Any]:
        with self.lock:
            if self.state.get("frozen"):
                return {"state": self.snapshot(), "output_dir": str(self.workdir / "frozen")}
            selected = select_final(self.items, self.state["annotations"], self.target)
            frozen_dir = self.workdir / "frozen"
            staging = self.workdir / ".frozen.tmp"
            if frozen_dir.exists() or staging.exists():
                raise ValueError("Frozen output already exists; inspect it before retrying")
            (staging / "images").mkdir(parents=True)
            for record in selected:
                source = self.image_dir / Path(record["filename"]).name
                if file_sha256(source) != record["sha256"]:
                    raise ValueError("Image hash changed before freeze")
                shutil.copy2(source, staging / "images" / Path(record["filename"]).name)
            with (staging / "blind_test_labels.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "order", "candidate_index", "filename", "code", "role", "sha256"
                ])
                writer.writeheader()
                writer.writerows(selected)
            skipped = []
            for index, item in enumerate(self.items, start=1):
                value = self.state["annotations"].get(item["filename"], {})
                if value.get("status") == "skipped":
                    skipped.append({
                        "candidate_index": index,
                        "filename": item["filename"],
                        "sha256": item["sha256"],
                        "role": item["role"],
                        "reason": value["reason"],
                    })
            reserve_used = sum(1 for item in selected if item["role"] == "reserve")
            frozen_manifest = {
                "schema_version": 1,
                "dataset": "captcha_blind_test_200",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "archive_sha256": self.archive_hash,
                "candidate_manifest_sha256": self.manifest_hash,
                "sampling_seed": self.manifest["seed"],
                "target_count": self.target,
                "labeled_count": len(selected),
                "skipped_count": len(skipped),
                "reserve_used": reserve_used,
                "reserve_remaining": len(self.manifest["reserve"]) - reserve_used,
                "ocr_predictions_used_for_annotation": False,
                "records": selected,
                "skipped": skipped,
            }
            (staging / "blind_test_manifest.json").write_text(
                json.dumps(frozen_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(staging, frozen_dir)
            self.state["frozen"] = True
            self.state["frozen_at_utc"] = frozen_manifest["frozen_at_utc"]
            atomic_json(self.state_path, self.state)
            return {"state": self.snapshot(), "output_dir": str(frozen_dir)}


class Handler(BaseHTTPRequestHandler):
    app: Application
    server_version = "emtl-OCR-LocalAnnotator/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[annotator] " + fmt % args)

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def local_host(self) -> bool:
        host = self.headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
        return host in {"127.0.0.1", "localhost"}

    def do_GET(self) -> None:
        if not self.local_host():
            self.send_json(403, {"error": "Only localhost requests are accepted"})
            return
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/state":
            with self.app.lock:
                self.send_json(200, self.app.snapshot())
        elif path.startswith("/api/image/"):
            filename = unquote(path[len("/api/image/"):])
            item = next((x for x in self.app.items if x["filename"] == filename), None)
            if item is None:
                self.send_json(404, {"error": "Unknown image"})
                return
            image = self.app.image_dir / Path(filename).name
            if not image.is_file() or file_sha256(image) != item["sha256"]:
                self.send_json(404, {"error": "Image missing or hash mismatch"})
                return
            body = image.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self.local_host():
            self.send_json(403, {"error": "Only localhost requests are accepted"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 10000:
                raise ValueError("Request body too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if self.path == "/api/annotation":
                self.send_json(200, {"state": self.app.annotate(payload)})
            elif self.path == "/api/freeze":
                self.send_json(200, self.app.freeze())
            else:
                self.send_json(404, {"error": "Not found"})
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": "Internal error: " + str(error)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/blind-test-candidates.json.gz"))
    parser.add_argument("--workdir", type=Path, default=Path(".local-data/p2-blind-test"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = Application(args.archive, args.manifest, args.workdir)
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = "http://127.0.0.1:%d/" % args.port
    print("Blind-test annotator: " + url)
    print("Local work directory: " + str(app.workdir))
    print("Candidates: %d (primary=%d, reserve=%d)" % (
        len(app.items), len(app.manifest["primary"]), len(app.manifest["reserve"])
    ))
    print("Progress: %d/%d labeled; frozen=%s" % (
        count_labeled(app.state), app.target, app.state.get("frozen", False)
    ))
    print("Bound to 127.0.0.1 only. Press Ctrl+C to stop; progress is saved automatically.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local annotator. Progress remains saved.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
