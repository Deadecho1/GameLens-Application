from pathlib import Path
from typing import List, Tuple

import torch
from decord import VideoReader, cpu


class VideoClipSampler:
    def open_video(self, video_path: str):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        fps = float(vr.get_avg_fps())
        duration = num_frames / fps
        return vr, num_frames, fps, duration

    def sample_window_frames(
        self,
        vr,
        start_frame: int,
        end_frame: int,
        num_frames: int,
    ) -> List:
        end_frame = max(end_frame, start_frame + 1)

        indices = torch.linspace(start_frame, end_frame - 1, steps=num_frames)
        indices = indices.round().long().clamp(start_frame, end_frame - 1)

        frames = vr.get_batch(indices.tolist()).asnumpy()
        return [frames[i] for i in range(frames.shape[0])]

    def get_frame(self, vr, frame_index: int):
        frame_index = max(0, min(frame_index, len(vr) - 1))
        return vr[frame_index].asnumpy()

    def get_frame_range(self, vr, start_frame: int, end_frame: int, step: int = 1) -> Tuple[List[int], List]:
        if len(vr) == 0:
            return [], []

        start_frame = max(0, start_frame)
        end_frame = min(len(vr) - 1, end_frame)
        step = max(1, step)

        if end_frame < start_frame:
            return [], []

        indices = list(range(start_frame, end_frame + 1, step))
        frames = vr.get_batch(indices).asnumpy()
        return indices, [frames[i] for i in range(frames.shape[0])]

    def list_mp4_files(self, input_dir: Path) -> List[Path]:
        return sorted(
            p for p in input_dir.glob("*.mp4")
            if p.is_file()
        )
