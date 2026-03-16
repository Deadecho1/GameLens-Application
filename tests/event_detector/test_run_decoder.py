"""Unit tests for RunDecoder — no ML model required."""
from __future__ import annotations

import pytest

from scripts.event_detector.config import DetectorConfig
from scripts.event_detector.labels import LABEL_START, LABEL_END, LABEL_CHOICE, LABEL_DROP
from scripts.event_detector.models import PeakEvent, RunCandidate
from scripts.event_detector.run_decoder import RunDecoder


def _event(label: str, time: float, score: float = 0.8) -> PeakEvent:
    return PeakEvent(
        label=label, time=time, score=score,
        start_time=time - 1.0, end_time=time + 1.0,
    )


def _empty_peaks() -> dict:
    return {LABEL_START: [], LABEL_END: [], LABEL_CHOICE: [], LABEL_DROP: []}


@pytest.fixture
def config() -> DetectorConfig:
    return DetectorConfig()


@pytest.fixture
def decoder(config) -> RunDecoder:
    return RunDecoder(config)


class TestShortRunPenalty:
    def test_no_penalty_for_long_run(self, decoder):
        assert decoder.short_run_penalty(60.0) == pytest.approx(0.0)

    def test_penalty_for_short_run(self, decoder):
        penalty = decoder.short_run_penalty(4.0)
        assert penalty > 0.0
        assert penalty <= 1.0

    def test_zero_duration_max_penalty(self, decoder):
        assert decoder.short_run_penalty(0.0) == pytest.approx(1.0)


class TestScoreRunCandidate:
    def test_end_before_start_returns_neg_inf(self, decoder):
        s = _event(LABEL_START, 10.0)
        e = _event(LABEL_END, 5.0)
        score = decoder.score_run_candidate(s, e, [], [], [], [])
        assert score == float("-inf")

    def test_basic_valid_run_has_positive_score(self, decoder):
        s = _event(LABEL_START, 5.0)
        e = _event(LABEL_END, 65.0)
        score = decoder.score_run_candidate(s, e, [s], [e], [], [])
        assert score > 0.0

    def test_inside_events_increase_score(self, decoder):
        s = _event(LABEL_START, 5.0)
        e = _event(LABEL_END, 65.0)
        choice = _event(LABEL_CHOICE, 30.0)
        score_without = decoder.score_run_candidate(s, e, [s], [e], [], [])
        score_with = decoder.score_run_candidate(s, e, [s], [e], [choice], [])
        assert score_with > score_without


class TestWeightedIntervalScheduling:
    def test_empty_input(self, decoder):
        assert decoder.weighted_interval_scheduling([]) == []

    def test_single_candidate_selected(self, decoder):
        s = _event(LABEL_START, 5.0)
        e = _event(LABEL_END, 65.0)
        candidate = RunCandidate(start_event=s, end_event=e, start_time=5.0, end_time=65.0, score=1.5)
        result = decoder.weighted_interval_scheduling([candidate])
        assert len(result) == 1

    def test_non_overlapping_candidates_both_selected(self, decoder):
        s1, e1 = _event(LABEL_START, 5.0), _event(LABEL_END, 60.0)
        s2, e2 = _event(LABEL_START, 70.0), _event(LABEL_END, 130.0)
        c1 = RunCandidate(start_event=s1, end_event=e1, start_time=5.0, end_time=60.0, score=1.5)
        c2 = RunCandidate(start_event=s2, end_event=e2, start_time=70.0, end_time=130.0, score=1.5)
        # Must sort by end_time first (as build_run_candidates does)
        candidates = sorted([c1, c2], key=lambda r: (r.end_time, r.start_time))
        result = decoder.weighted_interval_scheduling(candidates)
        assert len(result) == 2

    def test_overlapping_candidates_best_selected(self, decoder):
        s1, e1 = _event(LABEL_START, 5.0), _event(LABEL_END, 60.0)
        s2, e2 = _event(LABEL_START, 30.0), _event(LABEL_END, 90.0)
        c1 = RunCandidate(start_event=s1, end_event=e1, start_time=5.0, end_time=60.0, score=2.0)
        c2 = RunCandidate(start_event=s2, end_event=e2, start_time=30.0, end_time=90.0, score=1.0)
        candidates = sorted([c1, c2], key=lambda r: (r.end_time, r.start_time))
        result = decoder.weighted_interval_scheduling(candidates)
        assert len(result) == 1
        assert result[0].score == pytest.approx(2.0)

    def test_negative_score_candidate_excluded(self, decoder):
        s = _event(LABEL_START, 5.0)
        e = _event(LABEL_END, 65.0)
        candidate = RunCandidate(start_event=s, end_event=e, start_time=5.0, end_time=65.0, score=-0.5)
        result = decoder.weighted_interval_scheduling([candidate])
        assert result == []


class TestDecodeRuns:
    def test_no_events_returns_empty(self, decoder):
        peaks = _empty_peaks()
        result = decoder.decode_runs(peaks)
        assert result == []

    def test_single_start_end_pair_creates_run(self, decoder):
        peaks = _empty_peaks()
        peaks[LABEL_START] = [_event(LABEL_START, 5.0, score=0.8)]
        peaks[LABEL_END] = [_event(LABEL_END, 65.0, score=0.8)]
        result = decoder.decode_runs(peaks)
        assert len(result) == 1
        assert result[0].start.label == LABEL_START
        assert result[0].end.label == LABEL_END

    def test_choices_assigned_to_run(self, decoder):
        peaks = _empty_peaks()
        peaks[LABEL_START] = [_event(LABEL_START, 5.0, score=0.8)]
        peaks[LABEL_END] = [_event(LABEL_END, 65.0, score=0.8)]
        peaks[LABEL_CHOICE] = [_event(LABEL_CHOICE, 30.0, score=0.7)]
        result = decoder.decode_runs(peaks)
        assert len(result) == 1
        assert len(result[0].choices) == 1
