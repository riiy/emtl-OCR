#!/usr/bin/env python3
"""Validate 200 manual labels and emit a hash-audited frozen CSV."""
import argparse,csv,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finalize(mp,pp,lp,fp):
 mb=Path(mp).read_bytes();m=json.loads(mb);allowed={x["id"]:x for x in m["selected"]+m["reserve"]}
 events=[json.loads(x) for x in Path(pp).read_text().splitlines() if x.strip()];seen=set();labels=[]
 for e in events:
  i=e.get("image_id")
  if i not in allowed or i in seen:raise ValueError("Unknown or duplicate image event")
  seen.add(i);item=allowed[i]
  if e.get("path")!=item["path"] or e.get("sha256")!=item["sha256"]:raise ValueError("Manifest mismatch")
  if e["status"]=="labeled":
   v=e.get("label")
   if not isinstance(v,str) or len(v)!=4 or not v.isascii() or not v.isdigit():raise ValueError("Label must be four ASCII digits")
   labels.append({"image_id":i,"path":item["path"],"sha256":item["sha256"],"label":v})
  elif e["status"]!="skipped":raise ValueError("Unknown status")
 if len(labels)!=m["target_labels"]:raise ValueError(f"Expected {m['target_labels']} labels; got {len(labels)}")
 labels.sort(key=lambda x:x["path"]);Path(lp).parent.mkdir(parents=True,exist_ok=True)
 with open(lp,"w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["image_id","path","sha256","label"]);w.writeheader();w.writerows(labels)
 result={"status":"FROZEN","label_count":len(labels),"skipped_count":sum(e["status"]=="skipped" for e in events),
  "manifest_sha256":hashlib.sha256(mb).hexdigest(),"progress_sha256":sha(pp),"labels_sha256":sha(lp)}
 Path(fp).parent.mkdir(parents=True,exist_ok=True);Path(fp).write_text(json.dumps(result,indent=2)+"\n")
 return result
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ("manifest","progress","labels","freeze"):p.add_argument("--"+x,required=True)
 a=p.parse_args();print(json.dumps(finalize(a.manifest,a.progress,a.labels,a.freeze),indent=2))
if __name__=="__main__":main()
