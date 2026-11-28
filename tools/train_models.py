
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_dataset import audit_dataset
from training.runner import MODEL_NAMES, SEEDS, configure_cpu_runtime, platform_name, train_one_run

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the approved P3 three-model, three-seed CPU experiment.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "p3")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p3-training.json")
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()

    if (args.model is None) != (args.seed is None):
        parser.error("--model and --seed must be supplied together for a single run")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit_dataset(args.dataset, verify_images=True, verify_labels=True, verify_splits=True)
    configure_cpu_runtime()
    current_platform = platform_name()
    selected = [(args.model, args.seed)] if args.model else [
        (model, seed) for seed in SEEDS for model in MODEL_NAMES
    ]

    for model_name, seed in selected:
        run_dir = args.output / current_platform / f"model-{model_name.lower()}" / f"seed-{seed}"
        print(f"Starting model={model_name} seed={seed} output={run_dir}", flush=True)
        summary = train_one_run(
            dataset_dir=args.dataset,
            output_dir=run_dir,
            model_name=model_name,
            seed=seed,
            config=config,
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
