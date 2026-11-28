from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.data import CaptchaDataset, epoch_indices
from training.models import MODEL_SPECS, TinyCaptchaCNN, parameter_count

SEEDS = (20261127, 20261128, 20261129)
MODEL_NAMES = ("A", "B", "C")

def configure_cpu_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        if "after parallel work has started" not in str(exc):
            raise
    torch.use_deterministic_algorithms(True)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

def platform_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"x86_64", "amd64"}:
        distro = platform.freedesktop_os_release().get("ID", "").lower()
        if distro != "ubuntu":
            raise RuntimeError(f"P3 formal Linux training requires Ubuntu x86_64; detected {distro or 'unknown'}")
        return "ubuntu-x86_64"
    if system == "darwin" and machine in {"x86_64", "amd64"}:
        version = tuple(int(part) for part in platform.mac_ver()[0].split(".")[:2] if part.isdigit())
        if version and version < (12, 0):
            raise RuntimeError(f"P3 Intel Mac requires macOS 12.0 or later; detected {platform.mac_ver()[0]}")
        return "macos-x86_64"
    raise RuntimeError(f"Unsupported P3 platform: {system}/{machine}; expected Ubuntu or Intel macOS x86_64")

def environment_info() -> dict[str, Any]:
    import PIL
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _metrics(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    label_count = 0
    correct_images = 0
    image_count = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, targets in loader:
            logits = model(images)
            loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_labels = targets.numel()
            batch_images = targets.shape[0]
            loss_total += float(loss.detach()) * batch_labels
            label_count += batch_labels
            predictions = logits.detach().argmax(dim=2)
            correct_images += int((predictions == targets).all(dim=1).sum())
            image_count += batch_images

    if label_count == 0 or image_count == 0:
        raise ValueError("Cannot calculate metrics for an empty data loader")
    return loss_total / label_count, correct_images / image_count

def train_one_run(
    *,
    dataset_dir: Path | str,
    output_dir: Path | str,
    model_name: str,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    model_name = model_name.upper()
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model_name!r}")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")

    configure_cpu_runtime()
    set_global_seed(seed)
    train_data = CaptchaDataset(dataset_dir, "train", augment=True, seed=seed)
    validation_data = CaptchaDataset(dataset_dir, "validation", augment=False, seed=seed)
    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    if batch_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("batch_size, max_epochs and early_stopping_patience must be positive")
    learning_rate = float(config["optimizer"]["learning_rate"])
    weight_decay = float(config["optimizer"]["weight_decay"])

    model = TinyCaptchaCNN(model_name).cpu()
    params = parameter_count(model)
    expected_params = MODEL_SPECS[model_name]["parameters"]
    if params != expected_params:
        raise AssertionError(f"Model {model_name} parameter count {params} != {expected_params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    output_dir.mkdir(parents=True)
    (output_dir / "run_status.json").write_text(
        json.dumps({"status": "running", "model_name": model_name, "seed": seed}, indent=2) + "\n",
        encoding="utf-8",
    )
    best_accuracy = -1.0
    best_checkpoint_accuracy = -1.0
    best_checkpoint_loss = float("inf")
    best_epoch = 0
    epochs_without_accuracy_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best.pt"
    start_time = time.perf_counter()

    try:
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.perf_counter()
            train_data.set_epoch(epoch)
            train_order = epoch_indices(len(train_data), seed, epoch)
            train_loader = DataLoader(
                train_data, batch_size=batch_size, sampler=train_order,
                num_workers=0, drop_last=False,
            )
            validation_loader = DataLoader(
                validation_data, batch_size=batch_size, shuffle=False,
                num_workers=0, drop_last=False,
            )

            train_loss, train_accuracy = _metrics(model, train_loader, optimizer)
            validation_loss, validation_accuracy = _metrics(model, validation_loader, None)
            epoch_seconds = time.perf_counter() - epoch_start
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": learning_rate,
                "epoch_seconds": epoch_seconds,
            })

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                epochs_without_accuracy_improvement = 0
            else:
                epochs_without_accuracy_improvement += 1

            checkpoint_improved = (
                validation_accuracy > best_checkpoint_accuracy
                or (validation_accuracy == best_checkpoint_accuracy and validation_loss < best_checkpoint_loss)
            )
            if checkpoint_improved:
                best_checkpoint_accuracy = validation_accuracy
                best_checkpoint_loss = validation_loss
                best_epoch = epoch
                torch.save({
                    "format": "emtl-ocr-p3-checkpoint-v1",
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "model_parameters": params,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                    "config": config,
                    "environment": environment_info(),
                }, checkpoint_path)

            if epochs_without_accuracy_improvement >= patience:
                break

        if not history or not checkpoint_path.is_file():
            raise RuntimeError("Training ended without a valid checkpoint")

        duration_seconds = time.perf_counter() - start_time
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        run_config = {
            "model_name": model_name,
            "seed": seed,
            "model_parameters": params,
            "config": config,
            "environment": environment_info(),
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = {
            "format": "emtl-ocr-p3-run-summary-v1",
            "model_name": model_name,
            "seed": seed,
            "model_parameters": params,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "validation_accuracy": best_checkpoint_accuracy,
            "validation_loss": best_checkpoint_loss,
            "duration_seconds": duration_seconds,
            "checkpoint": "best.pt",
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "platform": platform_name(),
            "environment": environment_info(),
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "run_status.json").write_text(
            json.dumps({"status": "completed", "model_name": model_name, "seed": seed}, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
    except Exception as exc:
        (output_dir / "run_status.json").write_text(
            json.dumps({
                "status": "failed",
                "model_name": model_name,
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
