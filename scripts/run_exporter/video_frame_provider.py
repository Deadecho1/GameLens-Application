import io
from pathlib import Path
from typing import Dict

from decord import VideoReader, cpu
from PIL import Image


class VideoFrameProvider:
    def __init__(self, video_dir: Path, image_format: str = "PNG"):
        self.video_dir = video_dir
        self.image_format = image_format
        self._cache: Dict[str, VideoReader] = {}

    def _get_video_path(self, video_name: str) -> Path:
        video_path = self.video_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Matching video not found: {video_path}")
        return video_path

    def _get_reader(self, video_name: str) -> VideoReader:
        if video_name not in self._cache:
            video_path = self._get_video_path(video_name)
            self._cache[video_name] = VideoReader(str(video_path), ctx=cpu(0))
        return self._cache[video_name]

    def get_frame_bytes(self, video_name: str, frame_index: int) -> bytes:
        vr = self._get_reader(video_name)

        if frame_index < 0 or frame_index >= len(vr):
            raise IndexError(
                f"Frame index {frame_index} out of range for video {video_name} "
                f"(num_frames={len(vr)})"
            )

        frame_np = vr[frame_index].asnumpy()
        image = Image.fromarray(frame_np)

        buffer = io.BytesIO()
        image.save(buffer, format=self.image_format)
        return buffer.getvalue()