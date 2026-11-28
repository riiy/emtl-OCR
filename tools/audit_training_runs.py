
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.models import MODEL_SPECS
from training.runner import MODEL_NAMES, SEEDS

def _read_run(path: Path, expected_platform: str) -> dict[str, Any]:
    status_path = path / "run_status.json"
    if not status_path.is_file():
        raise ValueError(f"Missing run status: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        detail = status.get("error", status.get("status", "unknown"))
        raise ValueError(f"Run is not complete: {path} ({detail})")

    summary_path = path / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("format") != "emtl-ocr-p3-run-summary-v1":
        raise ValueError(f"Unsupported run summary: {summary_path}")
    if summary.get("platform") != expected_platform:
        raise ValueError(f"Platform mismatch in {summary_path}")
    model_name, seed = summary["model_name"], int(summary["seed"])
    if model_name not in MODEL_NAMES or seed not in SEEDS:
        raise ValueError(f"Unexpected model/seed in {summary_path}")
    if summary["model_parameters"] != MODEL_SPECS[model_name]["parameters"]:
        raise ValueError(f"Parameter count mismatch in {summary_path}")
    if not (path / "metrics.csv").is_file() or not (path / "run_config.json").is_file():
        raise ValueError(f"Incomplete run artifacts in {path}")
    with (path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        metric_rows = list(csv.DictReader(stream))
    if len(metric_rows) != int(summary["epochs_completed"]):
        raise ValueError(f"Epoch metric count mismatch in {path}")
    if not 1 <= int(summary["best_epoch"]) <= int(summary["epochs_completed"]):
        raise ValueError(f"Invalid best epoch in {path}")
    checkpoint = path / summary["checkpoint"]
    if not checkpoint.is_file():
        raise ValueError(f"Missing checkpoint: {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != summary["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint}")
    accuracy = float(summary["validation_accuracy"])
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"Invalid validation accuracy in {summary_path}")
    return summary | {"run_dir": str(path), "checkpoint_path": str(checkpoint)}

def _summarize_platform(root: Path, platform_name: str) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        model_rows = []
        for seed in SEEDS:
            run_path = root / platform_name / f"model-{model_name.lower()}" / f"seed-{seed}"
            if not run_path.is_dir():
                raise ValueError(f"Missing expected run: {run_path}")
            model_rows.append(_read_run(run_path, platform_name))
        accuracies = [float(row["validation_accuracy"]) for row in model_rows]
        by_model[model_name] = {
            "mean_validation_accuracy": statistics.mean(accuracies),
            "sample_std_validation_accuracy": statistics.stdev(accuracies),
            "mean_validation_loss": statistics.mean(float(row["validation_loss"]) for row in model_rows),
            "parameter_count": MODEL_SPECS[model_name]["parameters"],
            "runs": model_rows,
        }

    best_mean = max(item["mean_validation_accuracy"] for item in by_model.values())
    eligible = [
        name for name, item in by_model.items()
        if best_mean - item["mean_validation_accuracy"] <= 0.01 + 1e-12
    ]
    winner = min(eligible, key=lambda name: (MODEL_SPECS[name]["parameters"], name))
    winner_runs = by_model[winner]["runs"]
    median_accuracy = sorted(float(row["validation_accuracy"]) for row in winner_runs)[1]
    median_candidates = [
        row for row in winner_runs
        if float(row["validation_accuracy"]) == median_accuracy
    ]
    median_run = min(median_candidates, key=lambda row: (float(row["validation_loss"]), int(row["seed"])))
    return {
        "platform": platform_name,
        "models": by_model,
        "best_mean_model": max(by_model, key=lambda name: by_model[name]["mean_validation_accuracy"]),
        "selected_model": winner,
        "median_validation_accuracy": median_accuracy,
        "selected_median_run": median_run,
    }

def audit(root: Path) -> dict[str, Any]:
    platforms = {}
    for name in ("ubuntu-x86_64", "macos-x86_64"):
        if (root / name).exists():
            platforms[name] = _summarize_platform(root, name)
    if not platforms:
        raise ValueError(f"No P3 platform runs found under {root}")

    cross_platform = {"checked": False, "passed": None, "comparisons": []}
    if set(platforms) == {"ubuntu-x86_64", "macos-x86_64"}:
        comparisons = []
        passed = True
        ubuntu = platforms["ubuntu-x86_64"]["models"]
        macos = platforms["macos-x86_64"]["models"]
        for model_name in MODEL_NAMES:
            for seed in SEEDS:
                left = next(row for row in ubuntu[model_name]["runs"] if row["seed"] == seed)
                right = next(row for row in macos[model_name]["runs"] if row["seed"] == seed)
                delta = abs(float(left["validation_accuracy"]) - float(right["validation_accuracy"]))
                comparisons.append({
                    "model": model_name,
                    "seed": seed,
                    "ubuntu_accuracy": left["validation_accuracy"],
                    "macos_accuracy": right["validation_accuracy"],
                    "absolute_difference": delta,
                })
                passed = passed and delta <= 0.01 + 1e-12
        cross_platform = {"checked": True, "passed": passed, "comparisons": comparisons}

    formal_selection = None
    if "ubuntu-x86_64" in platforms:
        ubuntu = platforms["ubuntu-x86_64"]
        formal_selection = {
            "source_platform": "ubuntu-x86_64",
            "selected_model": ubuntu["selected_model"],
            "selected_median_run": ubuntu["selected_median_run"],
        }

    return {
        "format": "emtl-ocr-p3-audit-v1",
        "platforms": platforms,
        "formal_selection": formal_selection,
        "cross_platform": cross_platform,
        "test_split_evaluated": False,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Audit P3 training outputs without reading the test split.")
    parser.add_argument("--root", type=Path, default=Path("outputs/p3"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0 if result["cross_platform"]["passed"] is not False else 1

if __name__ == "__main__":
    raise SystemExit(main())
