from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv()

import cv2

from app_core.model_manager import ModelManager


def sample_frame(
    video_path: Path,
    frame_index: int | None = None,
    timestamp_sec: float | None = None,
) -> tuple[cv2.typing.MatLike, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if timestamp_sec is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
    elif frame_index is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    else:
        target = max(0, total // 2)
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()

    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to read a frame from the video.")

    return frame, total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample one frame from a video and run the choice classifier pipeline."
    )
    parser.add_argument(
        "--video",
        default="tests/clip18.mp4",
        help="Path to input .mp4 file.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index to sample. Defaults to middle frame.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Timestamp in seconds to sample. Overrides --frame.",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/output",
        help="Directory to save the sampled frame.",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, total = sample_frame(video_path, args.frame, args.time)

    frame_name = f"{video_path.stem}_frame.png"
    frame_path = out_dir / frame_name
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"Failed to write frame image to {frame_path}")

    print(f"Sampled frame saved to: {frame_path}")
    if total > 0:
        print(f"Video frames: {total}")

    extractor = ModelManager.get_choice_extractor()
    image_bytes = frame_path.read_bytes()
    result = extractor.extract(image_bytes)

    choices = result.get("choices", [])
    selected_choice = result.get("selected_choice", "")

    print(f"Choice options: {len(choices)}")
    if choices:
        for i, c in enumerate(choices, start=1):
            print(f"  {i}. {c}")
    print(f"Selected choice: {selected_choice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
