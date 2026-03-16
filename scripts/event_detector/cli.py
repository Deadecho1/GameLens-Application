import argparse
import logging
import os
from pathlib import Path

from app_core.config import AppConfig
from app_core.logging import configure_logging
from app_core.model_manager import ModelManager

from .config import DetectorConfig
from .folder_processor import FolderProcessor, VideoEventDetector
from .model_backend import ModelBackend
from .video_utils import VideoClipSampler


def _build_detector(app_config: AppConfig, detector_config: DetectorConfig) -> VideoEventDetector:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    bundle = ModelManager.load_event_detector(app_config.event_detector_model_dir)
    backend = ModelBackend(bundle)
    sampler = VideoClipSampler()
    return VideoEventDetector(config=detector_config, backend=backend, sampler=sampler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing .mp4 videos",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where 1 JSON per video will be written",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs",
    )
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    app_config = AppConfig.load()
    detector_config = DetectorConfig()
    detector = _build_detector(app_config, detector_config)
    processor = FolderProcessor(detector)
    processor.process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
