from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

IMAGE_SIZE = (64, 24)

def _sample_seed(seed: int, epoch: int, image_id: str) -> int:
    payload = f"{seed}:{epoch}:{image_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)

def _to_tensor(image: Image.Image) -> torch.Tensor:
    resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    pixels = (255.0 - pixels) / 255.0
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))

def preprocess_image(image: Image.Image) -> torch.Tensor:
    return _to_tensor(image)

def _translate(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return image
    shifted = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., :dy, :] = 0.0
    elif dy < 0:
        shifted[..., dy:, :] = 0.0
    if dx > 0:
        shifted[..., :, :dx] = 0.0
    elif dx < 0:
        shifted[..., :, dx:] = 0.0
    return shifted

def augment_image(image: Image.Image, seed: int, epoch: int, image_id: str) -> torch.Tensor:
    rng = random.Random(_sample_seed(seed, epoch, image_id))
    enhanced = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    if rng.random() < 0.5:
        enhanced = ImageEnhance.Brightness(enhanced).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.5:
        enhanced = ImageEnhance.Contrast(enhanced).enhance(rng.uniform(0.9, 1.1))

    values = _to_tensor(enhanced)
    if rng.random() < 0.3:
        sigma = rng.uniform(0.0, 0.02)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_sample_seed(seed, epoch, image_id) ^ 0x5DEECE66D)
        noise = torch.randn(values.shape, generator=generator, dtype=values.dtype)
        values = torch.clamp(values + noise * sigma, 0.0, 1.0)

    if rng.random() < 0.5:
        values = _translate(values, rng.choice((-1, 0, 1)), rng.choice((-1, 0, 1)))
    return values

class CaptchaDataset(Dataset):
    def __init__(self, dataset_dir: Path | str, split: str, *, augment: bool = False, seed: int = 20261127):
        if split not in {"train", "validation"}:
            raise ValueError("P3 training data may only use train or validation split")
        if augment and split != "train":
            raise ValueError("Random augmentation is only permitted for the train split")

        self.dataset_dir = Path(dataset_dir)
        manifest = json.loads((self.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        self.records = [record for record in manifest["images"] if record["split"] == split]
        if not self.records:
            raise ValueError(f"The {split!r} split is empty")

        labels = {}
        required_ids = {record["image_id"] for record in self.records}
        with (self.dataset_dir / "labels.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["image_id", "label"]:
                raise ValueError("labels.csv must contain exactly image_id,label")
            for row in reader:
                image_id = row["image_id"]
                if image_id not in required_ids:
                    continue
                label = row["label"]
                if len(label) != 4 or not label.isascii() or not label.isdigit():
                    raise ValueError(f"Invalid four-digit label for {image_id}")
                labels[image_id] = label

        missing = [record["image_id"] for record in self.records if record["image_id"] not in labels]
        if missing:
            raise ValueError(f"Missing labels for {len(missing)} images in {split}")
        self.labels = labels
        self.split = split
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_id = record["image_id"]
        image_path = self.dataset_dir / record["path"]
        with Image.open(image_path) as source:
            if self.augment:
                image_tensor = augment_image(source, self.seed, self.epoch, image_id)
            else:
                image_tensor = preprocess_image(source)
        target = torch.tensor([int(char) for char in self.labels[image_id]], dtype=torch.long)
        return image_tensor, target

def epoch_indices(length: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_sample_seed(seed, epoch, "epoch-order"))
    return torch.randperm(length, generator=generator).tolist()
