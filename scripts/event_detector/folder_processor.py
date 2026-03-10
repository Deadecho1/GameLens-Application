from pathlib import Path
from typing import List, Tuple

from .config import DetectorConfig
from .peak_detection import PeakDetector
from .refiner import EventRefiner
from .run_decoder import RunDecoder
from .serializer import JsonSerializer
from .video_utils import VideoClipSampler
from .model_backend import ModelBackend
from .models import DecodedRun, WindowResult


class VideoEventDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.backend = ModelBackend()
        self.sampler = VideoClipSampler()
        self.peak_detector = PeakDetector(self.config)
        self.run_decoder = RunDecoder(self.config)
        self.refiner = EventRefiner(self.config, self.backend, self.sampler)

    def collect_window_results(self, video_path: str, verbose: bool = False):
        vr, total_frames, fps, duration = self.sampler.open_video(video_path)

        window_frames = max(1, int(round(self.config.window_seconds * fps)))
        stride_frames = max(1, int(round(self.config.stride_seconds * fps)))

        if verbose:
            print(f"Video: {video_path}")
            print(f"FPS: {fps:.3f}")
            print(f"Duration: {duration:.3f}s")
            print(f"Total frames: {total_frames}")
            print(f"Model frames per clip: {self.backend.num_model_frames}")
            print()

        windows: List[WindowResult] = []

        for start_frame in range(0, total_frames - window_frames + 1, stride_frames):
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
                print(
                    f"[{wr.start_time:.2f}s - {wr.end_time:.2f}s] "
                    f"center={wr.center_time:.2f}s "
                    f"-> {best_label} ({best_score:.3f})"
                )

        return windows, vr, fps, duration

    def detect_video(self, video_path: str, verbose: bool = False) -> Tuple[List[DecodedRun], float, float]:
        windows, vr, fps, duration = self.collect_window_results(video_path, verbose=verbose)
        peaks = self.peak_detector.build_peaks(windows)
        decoded_runs = self.run_decoder.decode_runs(peaks)

        seen_ids = set()
        kept_events = []

        for run in decoded_runs:
            for ev in [run.start, run.end] + run.choices + run.drops:
                if id(ev) not in seen_ids:
                    seen_ids.add(id(ev))
                    kept_events.append(ev)

        for ev in kept_events:
            self.refiner.refine_event(
                event=ev,
                vr=vr,
                fps=fps,
                duration=duration,
            )

        for run in decoded_runs:
            run.choices = self.refiner.filter_events_by_final_threshold(run.choices)
            run.drops = self.refiner.filter_events_by_final_threshold(run.drops)

        return decoded_runs, fps, duration


class FolderProcessor:
    def __init__(self, detector: VideoEventDetector):
        self.detector = detector
        self.sampler = VideoClipSampler()
        self.serializer = JsonSerializer()

    def process_folder(self, input_dir: Path, output_dir: Path, verbose: bool = False) -> None:
        videos = self.sampler.list_mp4_files(input_dir)

        if not videos:
            print(f"No .mp4 files found in: {input_dir}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Device: {self.detector.backend.device}")
        print(f"Found {len(videos)} video(s).\n")

        ok = 0
        failed = 0

        for idx, video_path in enumerate(videos, start=1):
            print(f"[{idx}/{len(videos)}] Processing: {video_path.name}")

            try:
                decoded_runs, fps, duration = self.detector.detect_video(
                    str(video_path),
                    verbose=verbose,
                )

                out_json = output_dir / f"{video_path.stem}.json"
                self.serializer.save_runs_json(
                    video_path=str(video_path),
                    fps=fps,
                    duration=duration,
                    decoded_runs=decoded_runs,
                    json_out_path=str(out_json),
                )

                print(f"Saved: {out_json}\n")
                ok += 1

            except Exception as e:
                print(f"FAILED: {video_path.name}")
                print(f"Reason: {e}\n")
                failed += 1

        print("Done.")
        print(f"Successful: {ok}")
        print(f"Failed: {failed}")