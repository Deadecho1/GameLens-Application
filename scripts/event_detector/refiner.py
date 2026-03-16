from __future__ import annotations

from bisect import bisect_left
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import DetectorConfig
from .labels import LABEL_CHOICE
from .models import PeakEvent, RefinedEvent


class EventRefiner:
    def __init__(self, config: DetectorConfig, backend, sampler):
        self.config = config
        self.backend = backend
        self.sampler = sampler

    @staticmethod
    def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0:
            return values
        window = max(1, int(window))
        if window == 1 or values.size == 1:
            return values.astype(np.float32, copy=False)
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(values, kernel, mode="same")

    @staticmethod
    def _frame_signature(frame: np.ndarray) -> np.ndarray:
        frame_f = frame.astype(np.float32)
        gray = 0.299 * frame_f[..., 0] + 0.587 * frame_f[..., 1] + 0.114 * frame_f[..., 2]

        h, w = gray.shape
        target_h = min(64, h)
        target_w = min(64, w)

        if target_h <= 0 or target_w <= 0:
            return gray.astype(np.float32)

        row_idx = np.linspace(0, h - 1, num=target_h).round().astype(np.int32)
        col_idx = np.linspace(0, w - 1, num=target_w).round().astype(np.int32)
        small = gray[row_idx][:, col_idx]

        mean = float(small.mean())
        std = float(small.std())
        if std > 1e-6:
            small = (small - mean) / std
        else:
            small = small - mean

        return small.astype(np.float32, copy=False)

    def _compute_adjacent_diffs(self, frames: List[np.ndarray]) -> np.ndarray:
        if len(frames) < 2:
            return np.zeros(0, dtype=np.float32)

        signatures = [self._frame_signature(frame) for frame in frames]
        diffs = [
            float(np.mean(np.abs(signatures[i + 1] - signatures[i])))
            for i in range(len(signatures) - 1)
        ]
        return np.asarray(diffs, dtype=np.float32)

    @staticmethod
    def _nearest_scored_index(sorted_frames: List[int], frame: int) -> int:
        if not sorted_frames:
            return -1
        pos = bisect_left(sorted_frames, frame)
        if pos <= 0:
            return 0
        if pos >= len(sorted_frames):
            return len(sorted_frames) - 1
        before = sorted_frames[pos - 1]
        after = sorted_frames[pos]
        if abs(after - frame) < abs(frame - before):
            return pos
        return pos - 1

    @staticmethod
    def _collect_choice_candidate_frames(
        start_frame: int,
        end_frame: int,
        coarse_frame: int,
        clip_step: int,
    ) -> List[int]:
        if end_frame < start_frame:
            return []

        frames = {max(start_frame, min(end_frame, coarse_frame))}

        left = coarse_frame
        while left >= start_frame:
            frames.add(left)
            left -= clip_step

        right = coarse_frame + clip_step
        while right <= end_frame:
            frames.add(right)
            right += clip_step

        return sorted(frames)

    @staticmethod
    def _candidate_sort_key(item: Tuple[float, int, float, float]) -> Tuple[int, float]:
        score, frame, _, _ = item
        return frame, score

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
    def _hillclimb_refine_event(
        self, event: PeakEvent, vr, fps: float, duration: float
    ) -> RefinedEvent:
        half = self.config.hillclimb_window_seconds / 2.0

        min_center = max(half, event.start_time + half)
        max_center = min(duration - half, event.end_time - half)

        if min_center > max_center:
            fallback_center = max(
                half,
                min(duration - half, (event.start_time + event.end_time) / 2.0),
            )

            best_score, best_start, best_end = self.score_window_at_center(
                center_time=fallback_center,
                label=event.label,
                vr=vr,
                fps=fps,
                duration=duration,
            )

            return RefinedEvent(
                source=event,
                refined_time=fallback_center,
                refined_frame=int(round(fallback_center * fps)),
                refined_score=best_score,
                refined_window_start=best_start,
                refined_window_end=best_end,
                refinement_method="hillclimb-fallback",
            )

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

        return RefinedEvent(
            source=event,
            refined_time=best_center,
            refined_frame=int(round(best_center * fps)),
            refined_score=best_score,
            refined_window_start=best_start,
            refined_window_end=best_end,
            refinement_method="hillclimb",
        )

    def _build_choice_retry_candidates(
        self,
        selected_frame: int,
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> Tuple[List[int], List[float]]:
        retry_frames: List[int] = []
        seen = set()

        for lookback in self.config.choice_retry_lookback_frames:
            frame = selected_frame - int(lookback)
            frame = max(start_frame, min(end_frame, frame))
            if frame in seen:
                continue
            seen.add(frame)
            retry_frames.append(frame)

        retry_times = [frame / fps for frame in retry_frames]
        return retry_frames, retry_times

    @torch.no_grad()
    def _score_choice_candidates(
        self,
        candidate_frames: List[int],
        vr,
        fps: float,
        duration: float,
        label: str,
    ) -> List[Tuple[float, int, float, float]]:
        scored: List[Tuple[float, int, float, float]] = []
        for frame in candidate_frames:
            score, window_start, window_end = self.score_window_at_center(
                center_time=frame / fps,
                label=label,
                vr=vr,
                fps=fps,
                duration=duration,
            )
            scored.append((score, frame, window_start, window_end))
        return scored

    def _pick_latest_valid_candidate(
        self,
        candidates: List[Tuple[float, int, float, float]],
        max_frame: int | None = None,
    ) -> Tuple[Tuple[float, int, float, float] | None, str | None]:
        filtered = candidates
        if max_frame is not None:
            filtered = [item for item in filtered if item[1] <= max_frame]

        acceptable = [
            item for item in filtered
            if item[0] >= self.config.choice_min_clip_score
        ]
        preferred = [
            item for item in acceptable
            if item[0] >= self.config.choice_preferred_clip_score
        ]

        if preferred:
            return max(preferred, key=self._candidate_sort_key), "preferred"
        if acceptable:
            return max(acceptable, key=self._candidate_sort_key), "acceptable"
        if filtered:
            return max(filtered, key=lambda x: x[0]), "score"
        return None, None

    def _adaptive_exit_threshold(self, smoothed: np.ndarray) -> float:
        if smoothed.size == 0:
            return float("inf")
        max_diff = float(np.max(smoothed))
        mean_diff = float(np.mean(smoothed))
        std_diff = float(np.std(smoothed))
        adaptive = mean_diff + 0.5 * std_diff
        relative = self.config.choice_exit_min_strength_ratio * max_diff
        return max(adaptive, relative)

    def _forward_exit_transition_frame(
        self,
        frame_indices: List[int],
        smoothed: np.ndarray,
        score_by_frame: Dict[int, float],
        coarse_frame: int,
    ) -> int | None:
        if not frame_indices or smoothed.size == 0 or not score_by_frame:
            return None

        scored_frames = sorted(score_by_frame.keys())
        scored_values = [score_by_frame[f] for f in scored_frames]
        confirm_needed = max(1, int(self.config.choice_exit_confirm_frames))
        threshold = self._adaptive_exit_threshold(smoothed)

        coarse_idx = min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - coarse_frame))
        max_diff_idx = len(frame_indices) - 2
        start_diff_idx = max(0, min(coarse_idx, max_diff_idx))

        for diff_idx in range(start_diff_idx, max_diff_idx + 1):
            diff_value = float(smoothed[diff_idx])
            if diff_value < threshold:
                continue

            pre_frame = frame_indices[diff_idx]
            pre_scored_idx = self._nearest_scored_index(scored_frames, pre_frame)
            if pre_scored_idx < 0:
                continue

            pre_score = scored_values[pre_scored_idx]
            if pre_score < self.config.choice_min_clip_score:
                continue

            post_scores: List[float] = []
            for j in range(pre_scored_idx + 1, min(len(scored_values), pre_scored_idx + 1 + confirm_needed)):
                post_scores.append(scored_values[j])

            if len(post_scores) < confirm_needed:
                continue

            if all(
                (post_score <= self.config.choice_min_clip_score)
                or ((pre_score - post_score) >= self.config.choice_exit_score_drop)
                for post_score in post_scores
            ):
                return pre_frame

        return None

    def _forward_score_collapse_frame(
        self,
        candidate_scores: List[Tuple[float, int, float, float]],
        coarse_frame: int,
    ) -> int | None:
        if not candidate_scores:
            return None

        sorted_candidates = sorted(candidate_scores, key=lambda item: item[1])
        forward_candidates = [item for item in sorted_candidates if item[1] >= coarse_frame]
        if not forward_candidates:
            return None

        confirm_needed = max(1, int(self.config.choice_exit_confirm_frames))
        running_peak = float("-inf")

        for idx, (score, frame, _, _) in enumerate(forward_candidates):
            running_peak = max(running_peak, float(score))
            if score < self.config.choice_min_clip_score:
                continue

            post_candidates = forward_candidates[idx + 1 : idx + 1 + confirm_needed]
            if len(post_candidates) < confirm_needed:
                continue

            if all(
                (post_score <= self.config.choice_min_clip_score)
                or ((running_peak - post_score) >= self.config.choice_exit_score_drop)
                for post_score, _, _, _ in post_candidates
            ):
                return int(frame)

        return None

    @torch.no_grad()
    def _refine_choice_event(
        self, event: PeakEvent, vr, fps: float, duration: float
    ) -> RefinedEvent:
        coarse_frame = int(round(event.time * fps))
        lookback_frames = max(0, int(round(self.config.choice_peak_lookback_seconds * fps)))
        after_pad_frames = max(0, int(round(self.config.choice_refine_region_pad_after_seconds * fps)))

        start_frame = max(0, int(round(event.start_time * fps)))
        end_frame = min(len(vr) - 1, int(round(event.end_time * fps)) + after_pad_frames)
        search_start_frame = max(start_frame, coarse_frame - lookback_frames)
        step = max(1, int(self.config.choice_frame_step))

        frame_indices, frames = self.sampler.get_frame_range(
            vr=vr,
            start_frame=search_start_frame,
            end_frame=end_frame,
            step=step,
        )

        if len(frame_indices) < 2:
            result = self._hillclimb_refine_event(event, vr, fps, duration)
            # Repackage with updated method name
            return RefinedEvent(
                source=result.source,
                refined_time=result.refined_time,
                refined_frame=result.refined_frame,
                refined_score=result.refined_score,
                refined_window_start=result.refined_window_start,
                refined_window_end=result.refined_window_end,
                retry_frames=result.retry_frames,
                retry_times=result.retry_times,
                refinement_method="choice-forward-fallback-hillclimb",
            )

        diffs = self._compute_adjacent_diffs(frames)
        smoothed = self._moving_average(diffs, self.config.choice_diff_smooth_window)

        clip_step = max(1, int(round(self.config.choice_clip_score_grid_step_seconds * fps)))
        candidate_frames = self._collect_choice_candidate_frames(
            start_frame=search_start_frame,
            end_frame=end_frame,
            coarse_frame=coarse_frame,
            clip_step=clip_step,
        )
        candidate_scores = self._score_choice_candidates(
            candidate_frames=candidate_frames,
            vr=vr,
            fps=fps,
            duration=duration,
            label=event.label,
        )
        score_by_frame = {frame: score for score, frame, _, _ in candidate_scores}

        exit_frame = self._forward_exit_transition_frame(
            frame_indices=frame_indices,
            smoothed=smoothed,
            score_by_frame=score_by_frame,
            coarse_frame=coarse_frame,
        )

        if exit_frame is not None:
            picked, picked_kind = self._pick_latest_valid_candidate(candidate_scores, max_frame=exit_frame)
            if picked is not None:
                best_score, best_frame, best_start, best_end = picked
                method = f"choice-forward-exit-last-{picked_kind}"
            else:
                best_score, best_frame, best_start, best_end = max(candidate_scores, key=lambda x: x[0])
                method = "choice-forward-exit-fallback-score"
        else:
            collapse_frame = self._forward_score_collapse_frame(
                candidate_scores=candidate_scores,
                coarse_frame=coarse_frame,
            )
            if collapse_frame is not None:
                forward_candidates = [
                    item for item in candidate_scores
                    if coarse_frame <= item[1] <= collapse_frame
                ]
                picked, picked_kind = self._pick_latest_valid_candidate(
                    forward_candidates,
                    max_frame=collapse_frame,
                )
                if picked is not None:
                    best_score, best_frame, best_start, best_end = picked
                    method = f"choice-forward-score-collapse-last-{picked_kind}"
                else:
                    best_score, best_frame, best_start, best_end = max(candidate_scores, key=lambda x: x[0])
                    method = "choice-forward-score-collapse-fallback-score"
            else:
                lookback_candidates = [
                    item for item in candidate_scores
                    if search_start_frame <= item[1] <= coarse_frame
                ]
                picked, picked_kind = self._pick_latest_valid_candidate(lookback_candidates, max_frame=coarse_frame)
                if picked is not None:
                    best_score, best_frame, best_start, best_end = picked
                    method = f"choice-lookback-fallback-last-{picked_kind}"
                elif candidate_scores:
                    best_score, best_frame, best_start, best_end = max(candidate_scores, key=lambda x: x[0])
                    method = "choice-lookback-fallback-score"
                else:
                    result = self._hillclimb_refine_event(event, vr, fps, duration)
                    return RefinedEvent(
                        source=result.source,
                        refined_time=result.refined_time,
                        refined_frame=result.refined_frame,
                        refined_score=result.refined_score,
                        refined_window_start=result.refined_window_start,
                        refined_window_end=result.refined_window_end,
                        retry_frames=result.retry_frames,
                        retry_times=result.retry_times,
                        refinement_method="choice-no-candidates-hillclimb",
                    )

        retry_frames, retry_times = self._build_choice_retry_candidates(
            selected_frame=best_frame,
            start_frame=frame_indices[0],
            end_frame=frame_indices[-1],
            fps=fps,
        )

        return RefinedEvent(
            source=event,
            refined_time=best_frame / fps,
            refined_frame=best_frame,
            refined_score=best_score,
            refined_window_start=best_start,
            refined_window_end=best_end,
            retry_frames=tuple(retry_frames),
            retry_times=tuple(retry_times),
            refinement_method=method,
        )

    @torch.no_grad()
    def refine_event(self, event: PeakEvent, vr, fps: float, duration: float) -> RefinedEvent:
        if event.label == LABEL_CHOICE:
            return self._refine_choice_event(event, vr, fps, duration)
        return self._hillclimb_refine_event(event, vr, fps, duration)

    def filter_events_by_final_threshold(self, events: List[RefinedEvent]) -> List[RefinedEvent]:
        out = []
        for ev in events:
            threshold = self.config.final_event_score_thresholds.get(ev.label, 0.0)
            if ev.refined_score >= threshold:
                out.append(ev)
        return out
