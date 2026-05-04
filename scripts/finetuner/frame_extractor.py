"""
frame_extractor.py — cv2-based frame extraction for YOLO boss-detector fine-tuning.

Reads source videos and writes JPEG frames into per-class subfolders
(boss/, regular_gameplay/) so the directory can be fed directly to
scripts/finetuner/yolo_cli.py.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2

from app_core.logging import get_logger

logger = get_logger(__name__)

BOSS_CLASS = "boss"
NONE_CLASS = "regular_gameplay"


def _open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, total_frames


def _existing_count(folder: Path, pattern: str) -> int:
    return len(list(folder.glob(pattern))) if folder.exists() else 0


def extract_boss_frames(
    video_path: Path,
    segments: list[dict],
    output_dir: Path,
    fps_sample: float = 1.0,
) -> int:
    """Extract frames from boss segments at fps_sample rate. Returns new frame count."""
    cap, video_fps, total_frames = _open_video(video_path)
    try:
        frame_step = max(1, round(video_fps / fps_sample))
        boss_dir = output_dir / BOSS_CLASS
        boss_dir.mkdir(parents=True, exist_ok=True)
        counter = _existing_count(boss_dir, "*.jpg")
        written = 0

        for seg in segments:
            start_frame = max(0, round(float(seg["start"]) * video_fps))
            end_frame = min(total_frames - 1, round(float(seg["end"]) * video_fps))
            frame_idx = start_frame
            while frame_idx <= end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break
                counter += 1
                cv2.imwrite(str(boss_dir / f"boss_{counter:04d}.jpg"), frame)
                written += 1
                frame_idx += frame_step

        return written
    finally:
        cap.release()


def extract_none_frames(
    video_path: Path,
    segments: list[dict],
    output_dir: Path,
    target_count: int,
    fps_sample: float = 1.0,
) -> int:
    """Sample target_count frames from outside all boss segments. Returns new frame count."""
    if target_count <= 0:
        return 0

    cap, video_fps, total_frames = _open_video(video_path)
    try:
        frame_step = max(1, round(video_fps / fps_sample))
        none_dir = output_dir / NONE_CLASS
        none_dir.mkdir(parents=True, exist_ok=True)

        candidates: list[int] = []
        frame_idx = 0
        while frame_idx < total_frames:
            t = frame_idx / video_fps
            if not any(float(s["start"]) <= t <= float(s["end"]) for s in segments):
                candidates.append(frame_idx)
            frame_idx += frame_step

        random.shuffle(candidates)
        to_sample = sorted(candidates[:target_count])

        counter = _existing_count(none_dir, "*.jpg")
        written = 0
        for fi in to_sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            counter += 1
            cv2.imwrite(str(none_dir / f"none_{counter:04d}.jpg"), frame)
            written += 1

        return written
    finally:
        cap.release()


def extract_all_frames(
    videos: list[dict],
    output_dir: Path,
    fps_sample: float = 1.0,
    none_multiplier: float = 2.0,
) -> dict[str, int]:
    """
    Extract boss and regular_gameplay frames from all videos.

    Each video entry: {path, duration, marks: [{type:"segment", start, end}]}
    Returns {boss: N, regular_gameplay: N}.
    """
    total_boss = 0
    # (video_path, segments, boss_count_for_this_video)
    per_video: list[tuple[Path, list[dict], int]] = []

    for video in videos:
        video_path = Path(video["path"])
        segments = [m for m in video.get("marks", []) if m.get("type") == "segment"]
        if not segments:
            continue
        logger.info("Extracting boss frames from %s (%d segments)", video_path.name, len(segments))
        boss_count = extract_boss_frames(video_path, segments, output_dir, fps_sample)
        logger.info("  wrote %d boss frames", boss_count)
        total_boss += boss_count
        per_video.append((video_path, segments, boss_count))

    none_target_total = int(total_boss * none_multiplier)
    total_none = 0

    for video_path, segments, boss_count in per_video:
        if total_boss > 0:
            video_none_target = max(1, round(none_target_total * boss_count / total_boss))
        else:
            video_none_target = 0
        logger.info("Extracting none frames from %s (target %d)", video_path.name, video_none_target)
        none_count = extract_none_frames(video_path, segments, output_dir, video_none_target, fps_sample)
        logger.info("  wrote %d none frames", none_count)
        total_none += none_count

    logger.info("Total: %d boss frames, %d regular_gameplay frames", total_boss, total_none)
    return {BOSS_CLASS: total_boss, NONE_CLASS: total_none}
