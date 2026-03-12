import argparse
from pathlib import Path

from app_core.model_manager import ModelManager

from .choice_service import ChoiceExtractionService
from .json_reader import EventJsonReader
from .run_exporter import RunExporter
from .video_frame_provider import VideoFrameProvider

from dotenv import load_dotenv

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Folder containing the per-video event JSON files",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Folder containing the matching .mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where 1 JSON per run will be written",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress logs",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON folder not found: {json_dir}")
    if not json_dir.is_dir():
        raise NotADirectoryError(f"JSON path is not a folder: {json_dir}")

    if not video_dir.exists():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Video path is not a folder: {video_dir}")

    choice_extractor = ModelManager.get_choice_extractor()
    choice_service = ChoiceExtractionService(choice_extractor)

    exporter = RunExporter(
        json_reader=EventJsonReader(),
        frame_provider=VideoFrameProvider(video_dir=video_dir),
        choice_service=choice_service,
    )

    exporter.process_folder(
        json_dir=json_dir,
        output_dir=output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()