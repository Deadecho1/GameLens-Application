"""End-to-end pipeline benchmark test.

Marked `slow` — excluded from normal test runs.
Run with:
    uv run pytest tests/benchmark/ -m slow -s

Requires:
  - ML model weights in models/event_detector/
  - Event Extraction service running at port 7761 (docker compose up -d)
  - ground_truth.json populated with expected results
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from .evaluate import evaluate_all, load_ground_truth

GROUND_TRUTH = Path(__file__).parent / "ground_truth.json"
TEST_VIDEOS = Path(__file__).parents[3] / "test_videos"


@pytest.mark.slow
def test_pipeline_accuracy(tmp_path: Path) -> None:
    event_dir = tmp_path / "events"
    run_dir = tmp_path / "runs"
    # Filtered video dir — only top-level MP4s; excludes fine_tune/ and any other subdirs
    video_dir = tmp_path / "videos"
    event_dir.mkdir()
    run_dir.mkdir()
    video_dir.mkdir()
    for mp4 in TEST_VIDEOS.glob("*.mp4"):
        (video_dir / mp4.name).symlink_to(mp4)

    # Stage 1: event detector
    subprocess.run(
        [
            sys.executable, "-m", "scripts.event_detector.cli",
            "--input-dir", str(video_dir),
            "--output-dir", str(event_dir),
        ],
        check=True,
    )

    # Stage 2: run exporter (calls live choice extraction service at port 7761)
    subprocess.run(
        [
            sys.executable, "-m", "scripts.run_exporter.cli",
            "--json-dir", str(event_dir),
            "--video-dir", str(video_dir),
            "--output-dir", str(run_dir),
        ],
        check=True,
    )

    gt = load_ground_truth(GROUND_TRUTH)
    summary = evaluate_all(gt, run_dir)
    summary.print_report()
    assert summary.all_passed(), "One or more benchmark metrics failed — see report above."
