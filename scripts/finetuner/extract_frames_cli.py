import argparse
import json
import logging
from pathlib import Path

from app_core.logging import configure_logging
from scripts.finetuner.frame_extractor import extract_all_frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract boss/regular_gameplay frames for YOLO fine-tuning."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON file: list of {path, duration, marks:[{type,start,end}]}",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for frame dataset")
    parser.add_argument("--fps-sample", type=float, default=1.0)
    parser.add_argument("--none-multiplier", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    input_json = Path(args.input_json)
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    videos = json.loads(input_json.read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = extract_all_frames(
        videos=videos,
        output_dir=output_dir,
        fps_sample=args.fps_sample,
        none_multiplier=args.none_multiplier,
    )
    print(f"Done. boss={summary.get('boss', 0)}, regular_gameplay={summary.get('regular_gameplay', 0)}")


if __name__ == "__main__":
    main()
