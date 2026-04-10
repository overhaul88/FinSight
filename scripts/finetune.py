"""CLI wrapper for FinSight QLoRA fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.finetune import FineTuningConfig, run_finetune


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FinSight QLoRA fine-tuning.")
    parser.add_argument("--train-path", default="data/finetune/train.json")
    parser.add_argument("--val-path", default="data/finetune/val.json")
    parser.add_argument("--output-dir", default="models/mistral-finsight")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dry-run", action="store_true", help="Validate the pipeline and emit adapter metadata only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = FineTuningConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_path=args.train_path,
        val_path=args.val_path,
    )
    result = run_finetune(config=config, dry_run=args.dry_run)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

