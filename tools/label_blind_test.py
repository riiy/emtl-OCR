#!/usr/bin/env python3
"""Offline Tkinter labeler. Displays images only; imports no OCR engine."""
import argparse,datetime,hashlib,io,json,os,tarfile
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image,ImageTk
def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
class App:
 def __init__(self,archive,manifest,progress):
  self.archive=tarfile.open(archive,"r:gz");self.members={m.name:m for m in self.archive.getmembers() if m.isfile()}
  self.m=json.loads(Path(manifest).read_text())
  if digest(archive)!=self.m["source_archive_sha256"]:raise ValueError("Archive SHA-256 mismatch")
  self.progress=Path(progress);self.events=[]
  if self.progress.exists():self.events=[json.loads(x) for x in self.progress.read_text().splitlines() if x.strip()]
  self.done={e["image_id"] for e in self.events};self.items=self.m["selected"]+self.m["reserve"]
  self.target=self.m["target_labels"];self.count=sum(e["status"]=="labeled" for e in self.events)
  self.root=tk.Tk();self.root.title("emtl-OCR — blind manual labeling");self.root.geometry("610x360")
  self.info=tk.StringVar();self.image=tk.Label(self.root);self.image.pack(pady=10)
  tk.Label(self.root,textvariable=self.info).pack()
  tk.Label(self.root,text="输入恰好四位数字，保留前导零；无法辨认时跳过。").pack(pady=8)
  self.entry=tk.Entry(self.root,font=("TkFixedFont",22),justify="center",width=8);self.entry.pack()
  self.entry.bind("<Return>",lambda e:self.submit());self.entry.bind("<KeyRelease>",self.clean)
  row=tk.Frame(self.root);row.pack(pady=12)
  tk.Button(row,text="提交 (Enter)",command=self.submit).pack(side="left",padx=5)
  tk.Button(row,text="跳过 (S)",command=self.skip).pack(side="left",padx=5)
  tk.Button(row,text="退出并保存",command=self.close).pack(side="left",padx=5)
  self.root.bind("s",lambda e:self.skip());self.root.protocol("WM_DELETE_WINDOW",self.close)
  self.current=None;self.next()
 def clean(self,event=None):
  v=self.entry.get()
  if len(v)>4 or any(c not in "0123456789" for c in v):
   self.entry.delete(0,tk.END);self.entry.insert(0,"".join(c for c in v if c in "0123456789")[:4])
 def append(self,status,label=None):
  x=self.current;e={"image_id":x["id"],"path":x["path"],"sha256":x["sha256"],"status":status,"label":label,
   "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
  self.progress.parent.mkdir(parents=True,exist_ok=True)
  with self.progress.open("a") as f:f.write(json.dumps(e)+"\n");f.flush();os.fsync(f.fileno())
  self.events.append(e);self.done.add(x["id"])
  if status=="labeled":self.count+=1
 def next(self):
  if self.count>=self.target:
   self.info.set("完成 200/200，进度已保存。");self.image.configure(image="",text="人工标注完成")
   self.entry.configure(state="disabled");return
  self.current=next((x for x in self.items if x["id"] not in self.done),None)
  if self.current is None:messagebox.showerror("候选耗尽",f"仅完成 {self.count}/200");self.close();return
  member=self.members.get(self.current["path"])
  if member is None:raise ValueError("Image missing from archive")
  data=self.archive.extractfile(member).read()
  if hashlib.sha256(data).hexdigest()!=self.current["sha256"]:raise ValueError("Image hash mismatch")
  im=Image.open(io.BytesIO(data)).convert("RGB").resize((404,144),Image.Resampling.NEAREST)
  self.photo=ImageTk.PhotoImage(im);self.image.configure(image=self.photo,text="")
  self.info.set(f"已标注 {self.count}/200；已跳过 {sum(e['status']=='skipped' for e in self.events)}")
  self.entry.configure(state="normal");self.entry.delete(0,tk.END);self.entry.focus_set()
 def submit(self):
  if self.current is None or self.count>=self.target:return
  v=self.entry.get()
  if len(v)!=4 or not v.isascii() or not v.isdigit():messagebox.showwarning("格式错误","请输入恰好四位 ASCII 数字，例如 0042");return
  self.append("labeled",v);self.next()
 def skip(self):
  if self.current is not None and self.count<self.target:self.append("skipped");self.next()
 def close(self):self.archive.close();self.root.destroy()
 def run(self):self.root.mainloop()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--archive",required=True);p.add_argument("--manifest",required=True)
 p.add_argument("--progress",default="datasets/private/blind-test-progress.jsonl");a=p.parse_args()
 App(a.archive,a.manifest,a.progress).run()
if __name__=="__main__":main()
