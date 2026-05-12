import argparse
import logging
from pathlib import Path

from app_core.logging import configure_logging
from scripts.finetuner.yolo_trainer import run_finetuning


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the YOLO boss detector on game-specific frames."
    )
    parser.add_argument(
        "--clips-dir",
        required=True,
        help="Folder containing boss/ and regular_gameplay/ frame subfolders",
    )
    parser.add_argument("--output-dir", required=True, help="Where to write the fine-tuned model")
    parser.add_argument("--base-model-path", required=True, help="Base YOLO .pt file to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists() or not clips_dir.is_dir():
        raise NotADirectoryError(f"--clips-dir not found or not a directory: {clips_dir}")

    run_finetuning(
        clips_dir=clips_dir,
        base_model_path=Path(args.base_model_path),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
