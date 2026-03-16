from typing import Dict, List

from .config import DetectorConfig
from .labels import EVENT_LABELS
from .models import PeakEvent, WindowResult


class PeakDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config

    def is_local_peak(self, values: List[float], idx: int) -> bool:
        left = max(0, idx - self.config.peak_neighborhood)
        right = min(len(values), idx + self.config.peak_neighborhood + 1)
        center_val = values[idx]

        for j in range(left, right):
            if j == idx:
                continue
            if values[j] > center_val:
                return False
        return True

    def extract_raw_peaks(self, windows: List[WindowResult], label: str) -> List[PeakEvent]:
        values = [w.scores[label] for w in windows]
        peaks: List[PeakEvent] = []

        for i, w in enumerate(windows):
            s = values[i]
            if s < self.config.raw_candidate_threshold:
                continue
            if not self.is_local_peak(values, i):
                continue

            peaks.append(
                PeakEvent(
                    label=label,
                    time=w.center_time,
                    score=s,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    support_windows=(w,),
                )
            )

        return peaks

    def merge_close_peaks(self, peaks: List[PeakEvent], merge_gap_seconds: float) -> List[PeakEvent]:
        if not peaks:
            return []

        peaks = sorted(peaks, key=lambda p: p.time)
        groups: List[List[PeakEvent]] = [[peaks[0]]]

        for p in peaks[1:]:
            if p.time - groups[-1][-1].time <= merge_gap_seconds:
                groups[-1].append(p)
            else:
                groups.append([p])

        merged: List[PeakEvent] = []

        for group in groups:
            best = max(group, key=lambda x: x.score)
            total_weight = sum(x.score for x in group)
            weighted_time = sum(x.time * x.score for x in group) / max(total_weight, 1e-8)

            group_start = min(x.start_time for x in group)
            group_end = max(x.end_time for x in group)

            merged.append(
                PeakEvent(
                    label=best.label,
                    time=weighted_time,
                    score=best.score,
                    start_time=max(0.0, group_start - self.config.merge_boundary_pad_seconds),
                    end_time=group_end + self.config.merge_boundary_pad_seconds,
                    support_windows=tuple(sw for g in group for sw in g.support_windows),
                )
            )

        return merged

    def build_peaks(self, windows: List[WindowResult]) -> Dict[str, List[PeakEvent]]:
        out: Dict[str, List[PeakEvent]] = {}
        for label in EVENT_LABELS:
            raw = self.extract_raw_peaks(windows, label)
            merged = self.merge_close_peaks(raw, self.config.merge_gap_seconds[label])
            out[label] = merged
        return out