from __future__ import annotations

from typing import Dict, List, Protocol, Tuple


class ClipScorer(Protocol):
    """Scores a list of video frames against event labels."""

    num_model_frames: int

    def score_clip(self, frames: list) -> Dict[str, float]:
        ...


class VideoSampler(Protocol):
    """Opens and samples frames from a video file."""

    def open_video(self, video_path: str) -> Tuple[object, int, float, float]:
        """Returns (video_reader_handle, num_frames, fps, duration)."""
        ...

    def sample_window_frames(
        self,
        vr: object,
        start_frame: int,
        end_frame: int,
        num_frames: int,
    ) -> list:
        ...

    def get_frame(self, vr: object, frame_index: int) -> object:
        ...

    def get_frame_range(
        self,
        vr: object,
        start_frame: int,
        end_frame: int,
        step: int = 1,
    ) -> Tuple[List[int], list]:
        ...

    def list_mp4_files(self, input_dir: object) -> list:
        ...
