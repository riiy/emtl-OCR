import csv,importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("freeze",ROOT/"tools/finalize_blind_test.py")
freeze=importlib.util.module_from_spec(spec);spec.loader.exec_module(freeze)
class FreezeTests(unittest.TestCase):
 def test_freeze_preserves_leading_zero(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);m=p/"manifest.json";progress=p/"progress.jsonl"
   m.write_text(json.dumps({"target_labels":2,"selected":[{"id":"a","path":"a.img","sha256":"a"},{"id":"b","path":"b.img","sha256":"b"}],"reserve":[]}))
   events=[{"image_id":"a","path":"a.img","sha256":"a","status":"labeled","label":"0042"},{"image_id":"b","path":"b.img","sha256":"b","status":"labeled","label":"9310"}]
   progress.write_text("".join(json.dumps(x)+"\n" for x in events))
   result=freeze.finalize(m,progress,p/"labels.csv",p/"freeze.json")
   with (p/"labels.csv").open() as f:rows=list(csv.DictReader(f))
   self.assertEqual(rows[0]["label"],"0042");self.assertEqual(result["label_count"],2)
 def test_freeze_rejects_incomplete_progress(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);m=p/"manifest.json";progress=p/"progress.jsonl"
   m.write_text(json.dumps({"target_labels":1,"selected":[{"id":"a","path":"a.img","sha256":"a"}],"reserve":[]}))
   progress.write_text(json.dumps({"image_id":"a","path":"a.img","sha256":"a","status":"skipped"})+"\n")
   with self.assertRaises(ValueError):freeze.finalize(m,progress,p/"labels.csv",p/"freeze.json")
if __name__=="__main__":unittest.main()
