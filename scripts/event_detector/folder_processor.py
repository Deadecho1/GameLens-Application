from pathlib import Path
from typing import Dict, List, Tuple

from app_core.logging import get_logger
from .config import DetectorConfig
from .model_backend import ModelBackend
from .models import DecodedRun, PeakEvent, RefinedEvent, RefinedRun, WindowResult
from .peak_detection import PeakDetector
from .refiner import EventRefiner
from .run_decoder import RunDecoder
from .serializer import JsonSerializer
from .video_utils import VideoClipSampler

logger = get_logger(__name__)


class VideoEventDetector:
    def __init__(
        self,
        config: DetectorConfig | None = None,
        backend: ModelBackend | None = None,
        sampler: VideoClipSampler | None = None,
    ):
        self.config = config or DetectorConfig()
        self.backend = backend
        self.sampler = sampler or VideoClipSampler()
        self.peak_detector = PeakDetector(self.config)
        self.run_decoder = RunDecoder(self.config)
        self.refiner = EventRefiner(self.config, self.backend, self.sampler)

    def collect_window_results(self, video_path: str, verbose: bool = False):
        vr, total_frames, fps, duration = self.sampler.open_video(video_path)

        window_frames = max(1, int(round(self.config.window_seconds * fps)))
        stride_frames = max(1, int(round(self.config.stride_seconds * fps)))

        if verbose:
            logger.debug("Video: %s", video_path)
            logger.debug("FPS: %.3f", fps)
            logger.debug("Duration: %.3fs", duration)
            logger.debug("Total frames: %d", total_frames)
            logger.debug("Model frames per clip: %d", self.backend.num_model_frames)

        total_windows = max(0, (total_frames - window_frames) // stride_frames + 1)
        if verbose:
            logger.debug("Total windows to process: %d", total_windows)

        windows: List[WindowResult] = []

        for win_idx, start_frame in enumerate(range(0, total_frames - window_frames + 1, stride_frames), start=1):
            end_frame = start_frame + window_frames

            frames = self.sampler.sample_window_frames(
                vr=vr,
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=self.backend.num_model_frames,
            )

            scores = self.backend.score_clip(frames)

            center_time = ((start_frame + end_frame) / 2.0) / fps
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]

            wr = WindowResult(
                start_time=start_frame / fps,
                end_time=end_frame / fps,
                center_time=center_time,
                scores=scores,
            )
            windows.append(wr)

            if verbose:
                logger.debug(
                    "[%d/%d] [%.2fs - %.2fs] center=%.2fs -> %s (%.3f)",
                    win_idx, total_windows,
                    wr.start_time, wr.end_time, wr.center_time, best_label, best_score,
                )

        return windows, vr, fps, duration

    def detect_video(
        self, video_path: str, verbose: bool = False
    ) -> Tuple[List[RefinedRun], float, float]:
        logger.debug("Stage 1/4: collecting window results for %s", video_path)
        windows, vr, fps, duration = self.collect_window_results(video_path, verbose=verbose)
        logger.debug("Stage 1/4: done — %d windows collected", len(windows))

        logger.debug("Stage 2/4: building peaks")
        peaks = self.peak_detector.build_peaks(windows)
        logger.debug("Stage 2/4: done — %d peaks", len(peaks))

        logger.debug("Stage 3/4: decoding runs")
        decoded_runs = self.run_decoder.decode_runs(peaks)
        logger.debug("Stage 3/4: done — %d runs decoded", len(decoded_runs))

        logger.debug("Stage 4/4: deduplicating and refining events")
        # Deduplicate events by content key
        seen_keys: set = set()
        kept_events: List[PeakEvent] = []
        for run in decoded_runs:
            for ev in [run.start, run.end] + list(run.choices) + list(run.drops):
                key = (ev.label, ev.time, ev.score)
                if key not in seen_keys:
                    seen_keys.add(key)
                    kept_events.append(ev)

        logger.debug("Stage 4/4: %d unique events to refine", len(kept_events))
        # Refine each unique event — returns new RefinedEvent (no mutation)
        refined_map: Dict[Tuple, RefinedEvent] = {}
        for ev_idx, ev in enumerate(kept_events, start=1):
            logger.debug("  refining event %d/%d: label=%s time=%.2f", ev_idx, len(kept_events), ev.label, ev.time)
            refined = self.refiner.refine_event(event=ev, vr=vr, fps=fps, duration=duration)
            refined_map[(ev.label, ev.time, ev.score)] = refined

        def get_refined(ev: PeakEvent) -> RefinedEvent:
            return refined_map[(ev.label, ev.time, ev.score)]

        # Rebuild runs with RefinedEvent objects and apply final threshold filtering
        refined_runs: List[RefinedRun] = []
        for run in decoded_runs:
            refined_choices = self.refiner.filter_events_by_final_threshold(
                [get_refined(ev) for ev in run.choices]
            )
            refined_drops = self.refiner.filter_events_by_final_threshold(
                [get_refined(ev) for ev in run.drops]
            )
            refined_runs.append(
                RefinedRun(
                    start=get_refined(run.start),
                    end=get_refined(run.end),
                    choices=tuple(refined_choices),
                    drops=tuple(refined_drops),
                )
            )

        logger.debug("Stage 4/4: done — %d refined runs", len(refined_runs))
        return refined_runs, fps, duration


class FolderProcessor:
    def __init__(self, detector: VideoEventDetector):
        self.detector = detector
        self.sampler = VideoClipSampler()
        self.serializer = JsonSerializer()

    def process_folder(self, input_dir: Path, output_dir: Path, verbose: bool = False) -> None:
        videos = self.sampler.list_mp4_files(input_dir)

        if not videos:
            logger.warning("No .mp4 files found in: %s", input_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Device: %s", self.detector.backend.device)
        logger.info("Found %d video(s).", len(videos))

        ok = 0
        failed = 0

        for idx, video_path in enumerate(videos, start=1):
            logger.info("[%d/%d] Processing: %s", idx, len(videos), video_path.name)

            try:
                refined_runs, fps, duration = self.detector.detect_video(
                    str(video_path),
                    verbose=verbose,
                )

                out_json = output_dir / f"{video_path.stem}.json"
                self.serializer.save_runs_json(
                    video_path=str(video_path),
                    fps=fps,
                    duration=duration,
                    refined_runs=refined_runs,
                    json_out_path=str(out_json),
                )

                logger.info("Saved: %s", out_json)
                ok += 1

            except Exception as e:
                logger.error("FAILED: %s — %s", video_path.name, e)
                failed += 1

        logger.info("Done. Successful: %d  Failed: %d", ok, failed)
