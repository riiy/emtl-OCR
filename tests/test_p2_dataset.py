from __future__ import annotations

import csv
import gzip
import hashlib
import json
import tarfile
import tempfile
import unittest
from io import BytesIO, StringIO
from pathlib import Path

from tools.prepare_dataset import blind_candidates_manifest, read_archive, train_validation_manifest
from tools.annotate_blind_test import Application, load_manifest


def jpeg(index: int) -> bytes:
    return b"\xff\xd8\xff" + ("synthetic-%d" % index).encode("ascii") + b"\xff\xd9"


def make_archive(path: Path, counts: dict[str, int], extras: int = 5) -> dict[str, bytes]:
    rows, images = [], {}
    index = 0
    for source, count in counts.items():
        for _ in range(count):
            data = jpeg(index)
            digest = hashlib.sha256(data).hexdigest()
            name = "images/" + digest + ".img"
            images[name] = data
            rows.append({
                "filename": name, "code": "%04d" % (index % 10000),
                "source": source, "sample_id": str(index + 1), "sha256": digest,
            })
            index += 1
    for _ in range(extras):
        data = jpeg(index)
        digest = hashlib.sha256(data).hexdigest()
        images["images/" + digest + ".img"] = data
        index += 1
    text = StringIO()
    writer = csv.DictWriter(text, fieldnames=["filename", "code", "source", "sample_id", "sha256"])
    writer.writeheader()
    writer.writerows(rows)
    with tarfile.open(path, "w:gz") as archive:
        for name, data in [("hard_labeling/train_1100.csv", text.getvalue().encode()), *images.items()]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    return images


class DatasetManifestTests(unittest.TestCase):
    def test_stratified_split_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.tgz"
            make_archive(path, {"ACCEPTED": 10, "REJECTED": 5, "SKIPPED_INVALID_FORMAT": 5})
            audit = read_archive(path)
            one = train_validation_manifest(audit["rows"], audit["archive_sha256"], 123)
            two = train_validation_manifest(audit["rows"], audit["archive_sha256"], 123)
            self.assertEqual(one, two)
            self.assertEqual((one["counts"]["train"], one["counts"]["validation"]), (16, 4))
            self.assertEqual(one["counts"]["by_source"]["ACCEPTED"], {
                "total": 10, "train": 8, "validation": 2
            })
            train = {x["filename"] for x in one["train"]}
            validation = {x["filename"] for x in one["validation"]}
            self.assertFalse(train & validation)
            self.assertEqual(len(train | validation), 20)
            self.assertNotIn("code", one["train"][0])

    def test_blind_sample_and_reserve_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.tgz"
            make_archive(path, {"ACCEPTED": 10}, extras=287)
            audit = read_archive(path)
            one = blind_candidates_manifest(audit["candidates"], audit["archive_sha256"], 456)
            two = blind_candidates_manifest(audit["candidates"], audit["archive_sha256"], 456)
            self.assertEqual(one, two)
            self.assertEqual((len(one["primary"]), len(one["reserve"])), (200, 87))
            primary = {x["filename"] for x in one["primary"]}
            reserve = {x["filename"] for x in one["reserve"]}
            self.assertFalse(primary & reserve)

    def test_archive_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.tgz"
            images = make_archive(path, {"ACCEPTED": 2}, extras=1)
            with tarfile.open(path, "r:gz") as archive:
                csv_bytes = archive.extractfile("hard_labeling/train_1100.csv").read()
                entries = [(m.name, archive.extractfile(m).read()) for m in archive.getmembers()
                           if m.isfile() and m.name.startswith("images/")]
            rows = list(csv.DictReader(csv_bytes.decode().splitlines()))
            rows[0]["sha256"] = "0" * 64
            text = StringIO()
            writer = csv.DictWriter(text, fieldnames=["filename", "code", "source", "sample_id", "sha256"])
            writer.writeheader()
            writer.writerows(rows)
            with tarfile.open(path, "w:gz") as archive:
                for name, data in [("hard_labeling/train_1100.csv", text.getvalue().encode()), *entries]:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, BytesIO(data))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                read_archive(path)


class AnnotationTests(unittest.TestCase):
    def test_skip_replacement_leading_zero_and_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "data.tgz"
            images = make_archive(archive, {"ACCEPTED": 1}, extras=4)
            records = [
                {"filename": name, "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in images.items()
            ]
            primary = [{**records[0], "role": "primary"}, {**records[1], "role": "primary"}]
            reserve = [{**records[i], "role": "reserve"} for i in range(2, 5)]
            manifest = {
                "schema_version": 1,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "seed": 9, "target_labeled_count": 2, "candidate_count": 5,
                "primary": primary, "reserve": reserve,
            }
            manifest_path = root / "blind.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            app = Application(archive, manifest_path, root / "work")
            app.annotate({"filename": primary[0]["filename"], "status": "labeled", "code": "0007"})
            app.annotate({"filename": primary[1]["filename"], "status": "skipped", "reason": "无法辨认"})
            app.annotate({"filename": reserve[0]["filename"], "status": "labeled", "code": "1204"})
            result = app.freeze()
            self.assertTrue(result["state"]["frozen"])
            output = Path(result["output_dir"])
            with (output / "blind_test_labels.csv").open(encoding="utf-8", newline="") as stream:
                labels = list(csv.DictReader(stream))
            self.assertEqual(len(labels), 2)
            self.assertEqual(labels[0]["code"], "0007")
            self.assertEqual(labels[1]["role"], "reserve")
            self.assertTrue((output / "images" / Path(primary[0]["filename"]).name).is_file())
            with self.assertRaisesRegex(ValueError, "frozen"):
                app.annotate({"filename": primary[0]["filename"], "status": "labeled", "code": "1111"})

    def test_invalid_labels_and_compressed_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "data.tgz"
            images = make_archive(archive, {"ACCEPTED": 1}, extras=1)
            records = [
                {"filename": name, "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in images.items()
            ]
            manifest = {
                "schema_version": 1,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "seed": 1, "target_labeled_count": 1, "candidate_count": 2,
                "primary": [{**records[0], "role": "primary"}],
                "reserve": [{**records[1], "role": "reserve"}],
            }
            compressed = root / "blind.json.gz"
            with compressed.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                    stream.write(json.dumps(manifest).encode())
            self.assertEqual(load_manifest(compressed), manifest)
            app = Application(archive, compressed, root / "work")
            with self.assertRaisesRegex(ValueError, "four digits"):
                app.annotate({"filename": records[0]["filename"], "status": "labeled", "code": "123"})


if __name__ == "__main__":
    unittest.main()
