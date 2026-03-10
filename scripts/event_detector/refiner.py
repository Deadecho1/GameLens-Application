from typing import List, Tuple

import torch

from .config import DetectorConfig
from .models import PeakEvent


class EventRefiner:
    def __init__(self, config: DetectorConfig, backend, sampler):
        self.config = config
        self.backend = backend
        self.sampler = sampler

    @torch.no_grad()
    def score_window_at_center(
        self,
        center_time: float,
        label: str,
        vr,
        fps: float,
        duration: float,
    ) -> Tuple[float, float, float]:
        half = self.config.hillclimb_window_seconds / 2.0

        center_time = max(half, min(duration - half, center_time))
        start_t = center_time - half
        end_t = center_time + half

        start_frame = int(round(start_t * fps))
        end_frame = int(round(end_t * fps))

        frames = self.sampler.sample_window_frames(
            vr=vr,
            start_frame=start_frame,
            end_frame=end_frame,
            num_frames=self.backend.num_model_frames,
        )

        scores = self.backend.score_clip(frames)
        return scores[label], start_t, end_t

    @torch.no_grad()
    def refine_event(self, event: PeakEvent, vr, fps: float, duration: float) -> PeakEvent:
        half = self.config.hillclimb_window_seconds / 2.0

        min_center = max(half, event.start_time + half)
        max_center = min(duration - half, event.end_time - half)

        if min_center > max_center:
            fallback_center = max(
                half,
                min(duration - half, (event.start_time + event.end_time) / 2.0)
            )

            best_score, best_start, best_end = self.score_window_at_center(
                center_time=fallback_center,
                label=event.label,
                vr=vr,
                fps=fps,
                duration=duration,
            )

            event.refined_time = fallback_center
            event.refined_frame = int(round(fallback_center * fps))
            event.refined_score = best_score
            event.refined_window_start = best_start
            event.refined_window_end = best_end
            return event

        grid_centers: List[float] = []
        t = min_center
        while t <= max_center + 1e-9:
            grid_centers.append(t)
            t += self.config.refine_grid_step_seconds

        if not grid_centers:
            grid_centers = [min_center]

        best_center = grid_centers[0]
        best_score, best_start, best_end = self.score_window_at_center(
            center_time=best_center,
            label=event.label,
            vr=vr,
            fps=fps,
            duration=duration,
        )

        for center in grid_centers[1:]:
            score, window_start, window_end = self.score_window_at_center(
                center_time=center,
                label=event.label,
                vr=vr,
                fps=fps,
                duration=duration,
            )
            if score > best_score:
                best_center = center
                best_score = score
                best_start = window_start
                best_end = window_end

        current = best_center
        current_score = best_score
        current_start = best_start
        current_end = best_end

        for _ in range(self.config.hillclimb_max_iters):
            left_center = max(min_center, current - self.config.hillclimb_step_seconds)
            right_center = min(max_center, current + self.config.hillclimb_step_seconds)

            left_score, left_start, left_end = self.score_window_at_center(
                center_time=left_center,
                label=event.label,
                vr=vr,
                fps=fps,
                duration=duration,
            )

            right_score, right_start, right_end = self.score_window_at_center(
                center_time=right_center,
                label=event.label,
                vr=vr,
                fps=fps,
                duration=duration,
            )

            if left_score > current_score and left_score >= right_score:
                current = left_center
                current_score = left_score
                current_start = left_start
                current_end = left_end
            elif right_score > current_score and right_score > left_score:
                current = right_center
                current_score = right_score
                current_start = right_start
                current_end = right_end
            else:
                break

            if current_score > best_score:
                best_center = current
                best_score = current_score
                best_start = current_start
                best_end = current_end

        event.refined_time = best_center
        event.refined_frame = int(round(best_center * fps))
        event.refined_score = best_score
        event.refined_window_start = best_start
        event.refined_window_end = best_end
        return event

    def filter_events_by_final_threshold(self, events: List[PeakEvent]) -> List[PeakEvent]:
        out = []
        for ev in events:
            score = ev.refined_score if ev.refined_score is not None else ev.score
            threshold = self.config.final_event_score_thresholds.get(ev.label, 0.0)
            if score >= threshold:
                out.append(ev)
        return out