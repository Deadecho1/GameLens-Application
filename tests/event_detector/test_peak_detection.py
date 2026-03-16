"""Unit tests for PeakDetector — no ML model required."""
from __future__ import annotations

import pytest

from scripts.event_detector.config import DetectorConfig
from scripts.event_detector.labels import LABEL_START, LABEL_END, LABEL_DROP, LABEL_CHOICE, NONE_LABEL
from scripts.event_detector.models import WindowResult
from scripts.event_detector.peak_detection import PeakDetector


def _make_window(center: float, scores: dict, stride: float = 2.0) -> WindowResult:
    return WindowResult(
        start_time=center - stride / 2,
        end_time=center + stride / 2,
        center_time=center,
        scores=scores,
    )


def _uniform_scores(label: str, value: float) -> dict:
    """All labels at 0.01 except the given label."""
    s = {LABEL_START: 0.01, LABEL_END: 0.01, LABEL_DROP: 0.01, LABEL_CHOICE: 0.01, NONE_LABEL: 0.01}
    s[label] = value
    return s


@pytest.fixture
def config() -> DetectorConfig:
    return DetectorConfig()


@pytest.fixture
def detector(config) -> PeakDetector:
    return PeakDetector(config)


class TestIsLocalPeak:
    def test_single_element_is_peak(self, detector):
        assert detector.is_local_peak([0.9], 0) is True

    def test_highest_value_in_window_is_peak(self, detector):
        values = [0.1, 0.5, 0.3]
        assert detector.is_local_peak(values, 1) is True

    def test_non_peak_returns_false(self, detector):
        values = [0.3, 0.5, 0.8]
        assert detector.is_local_peak(values, 1) is False

    def test_tie_is_not_peak(self, detector):
        # Equal neighbor is not strictly greater, so center is a peak
        values = [0.5, 0.5, 0.3]
        # idx=0: neighbor at 1 is equal (0.5 > 0.5 is False) → is peak
        assert detector.is_local_peak(values, 0) is True

    def test_strictly_greater_neighbor_breaks_peak(self, detector):
        values = [0.8, 0.5, 0.3]
        assert detector.is_local_peak(values, 1) is False


class TestExtractRawPeaks:
    def test_no_peaks_below_threshold(self, detector):
        windows = [_make_window(t, _uniform_scores(LABEL_START, 0.1)) for t in [2.0, 4.0, 6.0]]
        peaks = detector.extract_raw_peaks(windows, LABEL_START)
        assert peaks == []

    def test_single_peak_detected(self, detector):
        windows = [
            _make_window(2.0, _uniform_scores(LABEL_START, 0.1)),
            _make_window(4.0, _uniform_scores(LABEL_START, 0.8)),
            _make_window(6.0, _uniform_scores(LABEL_START, 0.1)),
        ]
        peaks = detector.extract_raw_peaks(windows, LABEL_START)
        assert len(peaks) == 1
        assert peaks[0].label == LABEL_START
        assert peaks[0].score == pytest.approx(0.8)

    def test_peak_carries_correct_window_bounds(self, detector):
        windows = [
            _make_window(2.0, _uniform_scores(LABEL_START, 0.1)),
            _make_window(4.0, _uniform_scores(LABEL_START, 0.8)),
            _make_window(6.0, _uniform_scores(LABEL_START, 0.1)),
        ]
        peaks = detector.extract_raw_peaks(windows, LABEL_START)
        assert peaks[0].start_time == pytest.approx(3.0)
        assert peaks[0].end_time == pytest.approx(5.0)


class TestMergeClosePeaks:
    def test_empty_input_returns_empty(self, detector):
        assert detector.merge_close_peaks([], merge_gap_seconds=5.0) == []

    def test_single_peak_unchanged(self, detector):
        from scripts.event_detector.models import PeakEvent
        ev = PeakEvent(label=LABEL_START, time=5.0, score=0.8, start_time=4.0, end_time=6.0)
        merged = detector.merge_close_peaks([ev], merge_gap_seconds=5.0)
        assert len(merged) == 1
        assert merged[0].score == pytest.approx(0.8)

    def test_close_peaks_merge_into_one(self, detector):
        from scripts.event_detector.models import PeakEvent
        peaks = [
            PeakEvent(label=LABEL_START, time=5.0, score=0.7, start_time=4.0, end_time=6.0),
            PeakEvent(label=LABEL_START, time=7.0, score=0.9, start_time=6.0, end_time=8.0),
        ]
        merged = detector.merge_close_peaks(peaks, merge_gap_seconds=5.0)
        assert len(merged) == 1
        assert merged[0].score == pytest.approx(0.9)  # best score

    def test_distant_peaks_stay_separate(self, detector):
        from scripts.event_detector.models import PeakEvent
        peaks = [
            PeakEvent(label=LABEL_START, time=5.0, score=0.7, start_time=4.0, end_time=6.0),
            PeakEvent(label=LABEL_START, time=20.0, score=0.9, start_time=19.0, end_time=21.0),
        ]
        merged = detector.merge_close_peaks(peaks, merge_gap_seconds=5.0)
        assert len(merged) == 2


class TestBuildPeaks:
    def test_returns_dict_with_all_event_labels(self, detector):
        windows = [_make_window(t, _uniform_scores(LABEL_START, 0.1)) for t in range(0, 30, 2)]
        result = detector.build_peaks(windows)
        assert set(result.keys()) == {LABEL_START, LABEL_END, LABEL_DROP, LABEL_CHOICE}
