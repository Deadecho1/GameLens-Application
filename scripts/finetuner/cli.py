import argparse
import logging
from pathlib import Path

from app_core.config import AppConfig
from app_core.logging import configure_logging

from .trainer import run_finetuning


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the X-CLIP event detector on game-specific clips using LoRA."
    )
    parser.add_argument(
        "--clips-dir",
        required=True,
        help="Folder containing labelled clips in start/, end/, choice/, none/ subfolders",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the fine-tuned model",
    )
    parser.add_argument(
        "--base-model-dir",
        default=None,
        help="Base model to fine-tune from (default: GAMELENS_EVENT_DETECTOR_MODEL_DIR)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=2,
        help="LoRA rank (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max training epochs (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists() or not clips_dir.is_dir():
        raise NotADirectoryError(f"--clips-dir not found or not a directory: {clips_dir}")

    output_dir = Path(args.output_dir)

    if args.base_model_dir is not None:
        base_model_dir = Path(args.base_model_dir)
    else:
        app_config = AppConfig.load()
        base_model_dir = app_config.event_detector_model_dir

    run_finetuning(
        clips_dir=clips_dir,
        base_model_dir=base_model_dir,
        output_dir=output_dir,
        lora_rank=args.lora_rank,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
