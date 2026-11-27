#!/usr/bin/env python3
"""Build deterministic source-stratified split and blind-test manifests."""
import argparse,csv,hashlib,io,json,random,tarfile
from pathlib import Path
from PIL import Image
CSV_PATH="hard_labeling/train_1100.csv"
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def build(archive,seed=20261127):
 archive_sha=sha(archive)
 with tarfile.open(archive,"r:gz") as tf:
  members={m.name:m for m in tf.getmembers() if m.isfile()}
  rows=list(csv.DictReader(io.StringIO(tf.extractfile(CSV_PATH).read().decode())))
  if len(rows)!=1100:raise ValueError("Expected 1100 label rows")
  refs={r["filename"] for r in rows}; names=sorted(n for n in members if n.startswith("images/") and n.endswith(".img"))
  candidates=sorted(set(names)-refs)
  if len(names)!=1387 or len(candidates)!=287:raise ValueError("Unexpected image inventory")
  records=[]
  for name in candidates:
   data=tf.extractfile(members[name]).read();digest=hashlib.sha256(data).hexdigest()
   if Path(name).stem!=digest:raise ValueError("Image path/hash mismatch: "+name)
   with Image.open(io.BytesIO(data)) as im:
    im.verify()
   with Image.open(io.BytesIO(data)) as im:
    if im.size!=(101,36) or im.mode!="RGB" or im.format!="JPEG":raise ValueError("Unexpected image properties: "+name)
   records.append({"id":digest[:16],"path":name,"sha256":digest})
  shuffled=records.copy();random.Random(seed).shuffle(shuffled)
  blind={"schema_version":1,"source_archive_sha256":archive_sha,"seed":seed,
   "method":"sorted candidates; random.Random(seed).shuffle; first 200 selected",
   "target_labels":200,"candidate_count":287,"selected":shuffled[:200],"reserve":shuffled[200:],
   "policy":{"manual_only":True,"four_ascii_digits":True,"preserve_leading_zero":True,"no_ocr_predictions":True}}
  groups={}
  for r in rows:groups.setdefault(r["source"],[]).append(r)
  train=[];val=[];counts={}
  for source,group in sorted(groups.items()):
   group=sorted(group,key=lambda r:r["filename"]);random.Random(seed+sum(source.encode())).shuffle(group)
   n=round(.8*len(group));counts[source]={"train":n,"validation":len(group)-n}
   for r in group[:n]:train.append({k:r[k] for k in ("filename","sha256","source","sample_id")})
   for r in group[n:]:val.append({k:r[k] for k in ("filename","sha256","source","sample_id")})
  split={"schema_version":1,"source_archive_sha256":archive_sha,"seed":seed,
   "method":"source-stratified 80/20 deterministic per-source shuffle","source_counts":counts,
   "train_count":len(train),"validation_count":len(val),"train":sorted(train,key=lambda r:r["filename"]),
   "validation":sorted(val,key=lambda r:r["filename"])}
 return blind,split
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--archive",required=True)
 p.add_argument("--blind-output",required=True);p.add_argument("--split-output",required=True)
 p.add_argument("--seed",type=int,default=20261127);a=p.parse_args()
 blind,split=build(a.archive,a.seed)
 for path,obj in ((a.blind_output,blind),(a.split_output,split)):
  Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj,indent=2)+"\n")
 print(json.dumps({"seed":a.seed,"selected":200,"reserve":87,"train":split["train_count"],"validation":split["validation_count"],"archive_sha256":blind["source_archive_sha256"]},indent=2))
if __name__=="__main__":main()
