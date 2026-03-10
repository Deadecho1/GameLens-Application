import argparse
from pathlib import Path

from .folder_processor import FolderProcessor, VideoEventDetector


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

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    detector = VideoEventDetector()
    processor = FolderProcessor(detector)
    processor.process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()