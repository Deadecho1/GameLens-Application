import argparse
import json
import logging
import sys
from pathlib import Path

from app_core.logging import configure_logging

from .extractor import VALID_LABELS, extract_all_clips


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract labelled clips from a video for fine-tuning the event detector."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the source MP4 video file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output folder; per-label subfolders are created here",
    )
    parser.add_argument(
        "--events-json",
        default=None,
        help='Path to a JSON file containing events: [{"time": 1.5, "label": "choice"}, ...]',
    )
    parser.add_argument(
        "--event",
        nargs=2,
        metavar=("TIME", "LABEL"),
        action="append",
        default=[],
        help="Inline event: --event 12.5 choice (repeatable)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Clip duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Collect events from JSON file
    events: list[dict] = []
    if args.events_json:
        json_path = Path(args.events_json)
        if not json_path.is_file():
            print(f"ERROR: Events JSON not found: {json_path}", file=sys.stderr)
            sys.exit(1)
        with json_path.open() as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            print("ERROR: Events JSON must be a list of {time, label} objects.", file=sys.stderr)
            sys.exit(1)
        events.extend(loaded)

    # Collect inline --event pairs
    for time_str, label in args.event:
        try:
            events.append({"time": float(time_str), "label": label})
        except ValueError:
            print(f"ERROR: Invalid time value '{time_str}' in --event.", file=sys.stderr)
            sys.exit(1)

    if not events:
        print("ERROR: No events supplied. Use --events-json or --event.", file=sys.stderr)
        sys.exit(1)

    # Validate labels
    bad = {ev.get("label") for ev in events} - VALID_LABELS
    if bad:
        print(
            f"ERROR: Unknown label(s): {sorted(bad)}. Must be one of: {sorted(VALID_LABELS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)

    try:
        summary = extract_all_clips(
            video_path=video_path,
            events=events,
            output_dir=output_dir,
            duration=args.duration,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    total = sum(summary.values())
    counts_str = "  ".join(f"{label}={n}" for label, n in sorted(summary.items()) if n > 0)
    print(f"Done. {total} clip{'s' if total != 1 else ''} written to {output_dir}  [{counts_str}]")


if __name__ == "__main__":
    main()
