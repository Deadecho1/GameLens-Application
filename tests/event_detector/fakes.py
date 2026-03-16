"""Test stubs implementing ClipScorer and VideoSampler protocols."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from scripts.event_detector.labels import LABEL_TEXTS


class ConstantScorer:
    """Returns a fixed score distribution for every clip."""

    num_model_frames: int = 8

    def __init__(self, scores: Dict[str, float]) -> None:
        self._scores = scores

    def score_clip(self, frames: list) -> Dict[str, float]:
        return {label: self._scores.get(label, 0.0) for label in LABEL_TEXTS}


class SequentialScorer:
    """Returns scores from a pre-loaded sequence, one per call."""

    num_model_frames: int = 8

    def __init__(self, sequence: List[Dict[str, float]]) -> None:
        self._iter: Iterator[Dict[str, float]] = iter(sequence)

    def score_clip(self, frames: list) -> Dict[str, float]:
        return next(self._iter)


def _blank_frame(h: int = 4, w: int = 4) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class FakeVideoSampler:
    """Returns blank frames; no real file I/O."""

    def __init__(self, num_frames: int = 300, fps: float = 30.0) -> None:
        self._num_frames = num_frames
        self._fps = fps
        self._duration = num_frames / fps

    def open_video(self, video_path: str) -> Tuple[object, int, float, float]:
        return self, self._num_frames, self._fps, self._duration

    def sample_window_frames(
        self, vr: object, start_frame: int, end_frame: int, num_frames: int
    ) -> list:
        return [_blank_frame() for _ in range(num_frames)]

    def get_frame(self, vr: object, frame_index: int) -> np.ndarray:
        return _blank_frame()

    def get_frame_range(
        self, vr: object, start_frame: int, end_frame: int, step: int = 1
    ) -> Tuple[List[int], List[np.ndarray]]:
        indices = list(range(start_frame, end_frame + 1, step))
        return indices, [_blank_frame() for _ in indices]

    def list_mp4_files(self, input_dir: object) -> List[Path]:
        return []

    def __len__(self) -> int:
        return self._num_frames
