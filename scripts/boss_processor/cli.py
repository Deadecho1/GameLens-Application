import argparse
import logging
from pathlib import Path

from app_core.config import AppConfig
from app_core.logging import configure_logging

from dotenv import load_dotenv


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Scan run JSON files for boss fights and enrich them with boss data."
    )
    parser.add_argument(
        "--run-json-dir",
        required=True,
        help="Folder containing per-run JSON files (output of run_exporter)",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Folder containing the matching video files",
    )
    parser.add_argument(
        "--boss-model",
        required=False,
        default=None,
        help="Path to the YOLO boss classifier .pt file (required when --boss-backend=yolo)",
    )
    parser.add_argument(
        "--boss-backend",
        choices=["yolo", "gemma"],
        default="yolo",
        help="Backend to use for boss-fight classification: 'yolo' (default) or 'gemma' (Gemma 4 via Google AI Studio)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress logs",
    )
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    run_json_dir = Path(args.run_json_dir)
    video_dir = Path(args.video_dir)

    if not run_json_dir.exists() or not run_json_dir.is_dir():
        raise FileNotFoundError(f"Run JSON folder not found: {run_json_dir}")
    if not video_dir.exists() or not video_dir.is_dir():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")

    from scripts.boss_detector.scanner import BossScanner
    from scripts.boss_extractor.extractor import BossNameExtractor
    from scripts.boss_processor.processor import BossProcessor
    from scripts.run_exporter.video_frame_provider import VideoFrameProvider

    config = AppConfig.load()

    if args.boss_backend == "gemma":
        from scripts.boss_detector.gemma_classifier import GemmaBossClassifier
        classifier = GemmaBossClassifier()
    else:
        if args.boss_model is None:
            parser.error("--boss-model is required when --boss-backend=yolo")
        boss_model_path = Path(args.boss_model)
        if not boss_model_path.exists():
            raise FileNotFoundError(f"Boss model not found: {boss_model_path}")
        from scripts.boss_detector.classifier import BossClassifier
        classifier = BossClassifier(str(boss_model_path))

    processor = BossProcessor(
        frame_provider=VideoFrameProvider(video_dir=video_dir),
        boss_scanner=BossScanner(classifier=classifier),
        boss_name_extractor=BossNameExtractor(base_url=config.classifier_base_url),
    )

    processor.process_folder(run_json_dir=run_json_dir, video_dir=video_dir)


if __name__ == "__main__":
    main()
