"""
extractor.py — cv2-based clip extraction for fine-tuning data preparation.

Reads a source video and writes short MP4 clips centred on caller-supplied
timestamps.  Output clips are placed in per-label subfolders so the directory
can be fed directly to scripts/finetuner/cli.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from app_core.logging import get_logger

logger = get_logger(__name__)

# Valid labels — must match FOLDER_TO_IDX in scripts/finetuner/trainer.py
VALID_LABELS = {"start", "end", "drop", "choice", "none"}


def extract_clip(
    video_path: Path,
    center_time: float,
    output_path: Path,
    duration: float = 2.0,
) -> int:
    """
    Extract a clip of `duration` seconds centred on `center_time` from the
    video at `video_path` and write it to `output_path`.

    Returns the number of frames written.
    Raises ValueError if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        half = duration / 2.0
        start_frame = max(0, round((center_time - half) * fps))
        end_frame = min(total_frames - 1, round((center_time + half) * fps))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_written = 0
        for _ in range(end_frame - start_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames_written += 1

        writer.release()
        return frames_written
    finally:
        cap.release()


def extract_all_clips(
    video_path: Path,
    events: list[dict],
    output_dir: Path,
    duration: float = 2.0,
) -> dict[str, int]:
    """
    Extract clips for a list of events and write them into per-label subfolders
    under `output_dir`.

    `events` is a list of dicts with keys:
        "time"  — float, seconds from the start of the video
        "label" — str, one of VALID_LABELS

    Clips are named ``{label}_{n:03d}.mp4`` starting from 001, counted
    independently per label.

    Returns a dict mapping each label to the number of clips written.
    Raises ValueError for unknown labels.
    """
    # Validate labels up front
    for ev in events:
        label = ev.get("label", "")
        if label not in VALID_LABELS:
            raise ValueError(
                f"Unknown label '{label}'. Must be one of: {sorted(VALID_LABELS)}"
            )

    # Count existing clips per label so new clips don't overwrite prior runs
    counters: dict[str, int] = {}
    for ev in events:
        label = ev["label"]
        if label not in counters:
            existing = sorted((output_dir / label).glob("*.mp4")) if (output_dir / label).is_dir() else []
            counters[label] = len(existing)

    summary: dict[str, int] = {ev["label"]: 0 for ev in events}

    for ev in events:
        label = ev["label"]
        center_time = float(ev["time"])
        counters[label] += 1
        n = counters[label]
        output_path = output_dir / label / f"{label}_{n:03d}.mp4"

        logger.info("Extracting %s clip %d at %.2fs → %s", label, n, center_time, output_path)
        frames = extract_clip(video_path, center_time, output_path, duration)
        logger.debug("  wrote %d frames", frames)
        summary[label] += 1

    return summary
