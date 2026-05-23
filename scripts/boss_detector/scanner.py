from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Protocol

from app_core.logging import get_logger

if TYPE_CHECKING:
    from scripts.run_exporter.video_frame_provider import VideoFrameProvider


class BossClassifierProtocol(Protocol):
    def classify_frame(self, image): ...


logger = get_logger(__name__)


@dataclass
class BossFightSegment:
    boss_class: str  # raw YOLO class name
    boss_names: list  # human-readable names from OpenAI (filled later)
    start_time: float
    end_time: float
    duration: float
    player_died: bool
    sample_frame_index: int  # highest-confidence frame index for OpenAI


@dataclass
class _FrameDetection:
    frame_index: int
    time: float
    class_name: str
    confidence: float


class BossScanner:
    def __init__(
        self,
        classifier: BossClassifierProtocol,
        confidence_threshold: float = 0.99,
        sample_stride: int = 30,
        min_duration_s: float = 2.0,
        gap_tolerance_frames: int = 90,
        died_tolerance_s: float = 30.0,
        exclude_classes: tuple[str, ...] = ("regular_gameplay",),
    ) -> None:
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.sample_stride = sample_stride
        self.min_duration_s = min_duration_s
        self.gap_tolerance_frames = gap_tolerance_frames
        self.died_tolerance_s = died_tolerance_s
        self.exclude_classes = exclude_classes

    def scan_time_range(
        self,
        frame_provider: VideoFrameProvider,
        video_name: str,
        start_time: float,
        end_time: float,
        fps: float,
        run_end_time: Optional[float] = None,
    ) -> List[BossFightSegment]:
        """Scan frames between start_time and end_time (in seconds) for boss fights."""
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        if run_end_time is None:
            run_end_time = end_time

        detections: List[_FrameDetection] = []

        frame_index = start_frame
        while frame_index <= end_frame:
            try:
                pil_image = frame_provider.get_frame_pil(video_name, frame_index)
            except IndexError:
                break

            result = self.classifier.classify_frame(pil_image)
            del pil_image

            if (
                result.confidence >= self.confidence_threshold
                and result.class_name not in self.exclude_classes
            ):
                detections.append(
                    _FrameDetection(
                        frame_index=frame_index,
                        time=frame_index / fps,
                        class_name=result.class_name,
                        confidence=result.confidence,
                    )
                )
                logger.debug(
                    "  boss frame %d: class=%s conf=%.3f",
                    frame_index,
                    result.class_name,
                    result.confidence,
                )

            frame_index += self.sample_stride

        if not detections:
            return []

        return self._build_segments(detections, run_end_time, fps)

    def _build_segments(
        self,
        detections: List[_FrameDetection],
        run_end_time: float,
        fps: float,
    ) -> List[BossFightSegment]:
        # Group detections into contiguous segments, allowing gaps up to gap_tolerance_frames.
        groups: List[List[_FrameDetection]] = []
        current_group: List[_FrameDetection] = [detections[0]]

        for det in detections[1:]:
            gap = det.frame_index - current_group[-1].frame_index
            if gap <= self.gap_tolerance_frames + self.sample_stride:
                current_group.append(det)
            else:
                groups.append(current_group)
                current_group = [det]
        groups.append(current_group)

        segments: List[BossFightSegment] = []

        for group in groups:
            start_time = group[0].time
            end_time = group[-1].time
            duration = end_time - start_time

            if duration < self.min_duration_s:
                logger.debug(
                    "  boss segment [%.1fs-%.1fs] too short (%.1fs), discarding",
                    start_time,
                    end_time,
                    duration,
                )
                continue

            # Majority-vote the class name within this segment.
            boss_class = Counter(d.class_name for d in group).most_common(1)[0][0]

            # Pick the highest-confidence frame as the sample for OpenAI.
            sample_det = max(group, key=lambda d: d.confidence)

            player_died = abs(end_time - run_end_time) <= self.died_tolerance_s

            segments.append(
                BossFightSegment(
                    boss_class=boss_class,
                    boss_names=[],
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    player_died=player_died,
                    sample_frame_index=sample_det.frame_index,
                )
            )
            logger.debug(
                "  boss segment: class=%s start=%.1fs end=%.1fs duration=%.1fs died=%s",
                boss_class,
                start_time,
                end_time,
                duration,
                player_died,
            )

        return segments
