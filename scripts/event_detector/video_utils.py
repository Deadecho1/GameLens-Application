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

    def list_mp4_files(self, input_dir: Path) -> List[Path]:
        return sorted(
            p for p in input_dir.glob("*.mp4")
            if p.is_file()
        )