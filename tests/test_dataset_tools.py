import csv,importlib.util,json,tempfile,tarfile,io,hashlib,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prepare=load("prepare_dataset","tools/prepare_dataset.py")
freeze=load("finalize_blind_test","tools/finalize_blind_test.py")
class DatasetTests(unittest.TestCase):
 def test_split_is_deterministic_and_stratified(self):
  with tempfile.TemporaryDirectory() as d:
   archive=Path(d)/"a.tgz";rows=["filename,code,source,sample_id,sha256"]
   with tarfile.open(archive,"w:gz") as tf:
    for source,count in (("ACCEPTED",10),("REJECTED",5),("SKIPPED_INVALID_FORMAT",5)):
     for i in range(count):
      data=f"{source}-{i}".encode();name="images/"+hashlib.sha256(data).hexdigest()+".img"
      info=tarfile.TarInfo(name);info.size=len(data);tf.addfile(info,io.BytesIO(data))
      rows.append(f"{name},{i:04d},{source},{i},{hashlib.sha256(data).hexdigest()}")
    data="\n".join(rows).encode();info=tarfile.TarInfo(prepare.CSV_PATH);info.size=len(data);tf.addfile(info,io.BytesIO(data))
   one=prepare.build(str(archive),7);two=prepare.build(str(archive),7)
   self.assertEqual(one,two);self.assertEqual(one[1]["train_count"],16);self.assertEqual(one[1]["validation_count"],4)
 def test_freeze_preserves_leading_zero_and_requires_target(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);m=p/"m.json";progress=p/"p.jsonl"
   m.write_text(json.dumps({"target_labels":2,"selected":[{"id":"a","path":"a.img","sha256":"a"},{"id":"b","path":"b.img","sha256":"b"}],"reserve":[]}))
   events=[{"image_id":"a","path":"a.img","sha256":"a","status":"labeled","label":"0042"},{"image_id":"b","path":"b.img","sha256":"b","status":"labeled","label":"9310"}]
   progress.write_text("".join(json.dumps(x)+"\n" for x in events))
   result=freeze.finalize(m,progress,p/"labels.csv",p/"freeze.json")
   with (p/"labels.csv").open() as f:rows=list(csv.DictReader(f))
   self.assertEqual(rows[0]["label"],"0042");self.assertEqual(result["label_count"],2)
   progress.write_text(json.dumps(events[0])+"\n")
   with self.assertRaises(ValueError):freeze.finalize(m,progress,p/"labels.csv",p/"freeze.json")
if __name__=="__main__":unittest.main()
