import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from app_core.config import AppConfig
from app_core.logging import configure_logging, get_logger

logger = get_logger(__name__)


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
    parser.add_argument(
        "--allow-llm-fallback",
        action="store_true",
        help=(
            "Allow automatic fallback to Gemma/OpenAI if YOLO backend initialization fails. "
            "Disabled by default to avoid accidental LLM usage."
        ),
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

    scanner_kwargs: dict = {}

    if args.boss_backend == "gemma":
        from scripts.boss_detector.gemma_classifier import GemmaBossClassifier

        classifier = GemmaBossClassifier()
        # Remote LLM classification is expensive; scan more sparsely.
        scanner_kwargs = {
            "confidence_threshold": 0.75,
            "sample_stride": 150,
            "gap_tolerance_frames": 300,
        }
    else:
        if args.boss_model is None:
            parser.error("--boss-model is required when --boss-backend=yolo")
        boss_model_path = Path(args.boss_model)
        if not boss_model_path.exists():
            raise FileNotFoundError(f"Boss model not found: {boss_model_path}")

        try:
            from scripts.boss_detector.classifier import BossClassifier

            classifier = BossClassifier(str(boss_model_path))
        except Exception as e:
            if args.allow_llm_fallback:
                logger.warning(
                    "YOLO boss backend unavailable (%s). Falling back to LLM backend because --allow-llm-fallback is enabled.",
                    e,
                )
                from scripts.boss_detector.gemma_classifier import GemmaBossClassifier

                classifier = GemmaBossClassifier()
                scanner_kwargs = {
                    "confidence_threshold": 0.75,
                    "sample_stride": 150,
                    "gap_tolerance_frames": 300,
                }
            else:
                hint = ""
                if "cv2" in str(e) and "IMREAD_COLOR" in str(e):
                    hint = (
                        " Detected a broken OpenCV import (missing cv2.IMREAD_COLOR). "
                        "Reinstall opencv-python/opencv-python-headless in this environment."
                    )
                raise RuntimeError(
                    "YOLO boss backend initialization failed. "
                    "LLM fallback is disabled by default."
                    f"{hint}"
                    " If you explicitly want fallback, pass --allow-llm-fallback."
                    f" Original error: {e}"
                ) from e

    processor = BossProcessor(
        frame_provider=VideoFrameProvider(video_dir=video_dir),
        boss_scanner=BossScanner(classifier=classifier, **scanner_kwargs),
        boss_name_extractor=BossNameExtractor(base_url=config.classifier_base_url),
    )

    processor.process_folder(run_json_dir=run_json_dir, video_dir=video_dir)


if __name__ == "__main__":
    main()
