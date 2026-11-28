from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

try:
    import numpy as np
    import pytest
    import torch
    from PIL import Image

    from training.data import CaptchaDataset, _translate, augment_image, epoch_indices, preprocess_image
    from training.models import MODEL_SPECS, TinyCaptchaCNN, parameter_count
    from training.runner import train_one_run
    from tools.audit_training_runs import audit
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "pytest", "torch"}:
        raise
    # The legacy unittest-only CI job intentionally does not install PyTorch.
    # The dedicated P3 smoke job installs environment.yml and runs this file with pytest.

def _write_dataset(root: Path, train_count: int = 2, validation_count: int = 2) -> Path:
    images = root / "images"
    images.mkdir(parents=True)
    records = []
    labels = []
    for split, count, offset in (
        ("train", train_count, 0),
        ("validation", validation_count, train_count),
    ):
        for index in range(count):
            image_id = f"{offset + index + 1:064x}"
            pixels = np.zeros((36, 101, 3), dtype=np.uint8)
            pixels[:, :, 0] = (offset + index + 1) * 20
            pixels[:, :, 1] = np.arange(101, dtype=np.uint8)[None, :]
            pixels[:, :, 2] = np.arange(36, dtype=np.uint8)[:, None]
            Image.fromarray(pixels, "RGB").save(images / f"{image_id}.jpg", quality=90)
            records.append({"image_id": image_id, "path": f"images/{image_id}.jpg", "split": split})
            labels.append((image_id, "0042" if index == 0 else "1234"))

    (root / "manifest.json").write_text(json.dumps({"images": records}), encoding="utf-8")
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "label"])
        writer.writerows(labels)
    return root

def test_model_parameter_counts_and_output_shape():
    for name, spec in MODEL_SPECS.items():
        model = TinyCaptchaCNN(name)
        assert parameter_count(model) == spec["parameters"]
        output = model(torch.zeros(2, 3, 24, 64))
        assert output.shape == (2, 4, 10)

def test_preprocess_inverse_normalization_and_shape():
    white = preprocess_image(Image.new("RGB", (101, 36), "white"))
    black = preprocess_image(Image.new("RGB", (101, 36), "black"))
    assert white.shape == (3, 24, 64)
    assert torch.all(white == 0)
    assert torch.all(black == 1)

def test_translation_uses_zero_fill():
    source = torch.ones(3, 4, 5)
    shifted = _translate(source, 1, 1)
    assert torch.all(shifted[:, 0, :] == 0)
    assert torch.all(shifted[:, :, 0] == 0)
    assert torch.all(shifted[:, 1:, 1:] == 1)

def test_augmentation_is_stateless_and_bounded():
    image = Image.new("RGB", (101, 36), (80, 120, 160))
    first = augment_image(image, 20261127, 3, "a" * 64)
    second = augment_image(image, 20261127, 3, "a" * 64)
    assert torch.equal(first, second)
    assert first.shape == (3, 24, 64)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0

def test_epoch_shuffle_is_reproducible():
    assert epoch_indices(20, 20261127, 1) == epoch_indices(20, 20261127, 1)
    assert epoch_indices(20, 20261127, 1) != epoch_indices(20, 20261127, 2)

def test_dataset_preserves_leading_zero_and_rejects_test_split(tmp_path):
    dataset_dir = _write_dataset(tmp_path)
    dataset = CaptchaDataset(dataset_dir, "train", augment=False)
    _, target = dataset[0]
    assert "".join(str(int(x)) for x in target.tolist()) == "0042"
    with pytest.raises(ValueError, match="train or validation"):
        CaptchaDataset(dataset_dir, "test")

def test_smoke_training_writes_auditable_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("training.runner.platform_name", lambda: "ubuntu-x86_64")
    dataset_dir = _write_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "run"
    config = {
        "training": {"batch_size": 2, "max_epochs": 1, "early_stopping_patience": 1},
        "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0001},
    }
    summary = train_one_run(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        model_name="A",
        seed=20261127,
        config=config,
    )
    assert summary["epochs_completed"] == 1
    assert summary["model_parameters"] == 83888
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "run_config.json").is_file()
    assert json.loads((output_dir / "run_status.json").read_text())["status"] == "completed"
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    restored = TinyCaptchaCNN("A")
    restored.load_state_dict(checkpoint["model_state_dict"])
    assert checkpoint["model_parameters"] == parameter_count(restored)
    with pytest.raises(FileExistsError):
        train_one_run(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            model_name="A",
            seed=20261127,
            config=config,
        )

def test_training_audit_selects_small_model_within_one_point(tmp_path):
    root = tmp_path / "outputs"
    platform = root / "ubuntu-x86_64"
    accuracies = {
        "A": [0.980, 0.981, 0.982],
        "B": [0.974, 0.975, 0.976],
        "C": [0.979, 0.980, 0.981],
    }
    for model_name, values in accuracies.items():
        for seed, accuracy in zip((20261127, 20261128, 20261129), values):
            run_dir = platform / f"model-{model_name.lower()}" / f"seed-{seed}"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "best.pt"
            checkpoint.write_bytes(f"{model_name}-{seed}".encode())
            summary = {
                "format": "emtl-ocr-p3-run-summary-v1",
                "model_name": model_name,
                "seed": seed,
                "model_parameters": MODEL_SPECS[model_name]["parameters"],
                "validation_accuracy": accuracy,
                "validation_loss": 0.1,
                "best_epoch": 1,
                "epochs_completed": 1,
                "checkpoint": "best.pt",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "platform": "ubuntu-x86_64",
            }
            (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "run_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (run_dir / "metrics.csv").write_text("epoch\n1\n", encoding="utf-8")
            (run_dir / "run_config.json").write_text("{}", encoding="utf-8")

    result = audit(root)
    assert result["platforms"]["ubuntu-x86_64"]["selected_model"] == "B"
    assert result["cross_platform"]["checked"] is False
    assert result["test_split_evaluated"] is False
