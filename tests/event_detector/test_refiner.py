"""Unit tests for EventRefiner using fakes — no GPU or real video required."""
from __future__ import annotations

import pytest

from scripts.event_detector.config import DetectorConfig
from scripts.event_detector.labels import LABEL_START, LABEL_CHOICE, LABEL_TEXTS, NONE_LABEL
from scripts.event_detector.models import PeakEvent, RefinedEvent
from scripts.event_detector.refiner import EventRefiner
from tests.event_detector.fakes import ConstantScorer, FakeVideoSampler


def _start_event(time: float = 5.0, score: float = 0.8) -> PeakEvent:
    return PeakEvent(
        label=LABEL_START, time=time, score=score,
        start_time=time - 2.0, end_time=time + 2.0,
    )


def _choice_event(time: float = 10.0, score: float = 0.75) -> PeakEvent:
    return PeakEvent(
        label=LABEL_CHOICE, time=time, score=score,
        start_time=time - 2.0, end_time=time + 2.0,
    )


def _make_scorer(label: str = LABEL_START, value: float = 0.85) -> ConstantScorer:
    scores = {l: 0.01 for l in LABEL_TEXTS}
    scores[label] = value
    return ConstantScorer(scores)


@pytest.fixture
def config() -> DetectorConfig:
    return DetectorConfig()


@pytest.fixture
def sampler() -> FakeVideoSampler:
    return FakeVideoSampler(num_frames=600, fps=30.0)


class TestHillclimbRefine:
    def test_returns_refined_event_not_same_object(self, config, sampler):
        scorer = _make_scorer()
        refiner = EventRefiner(config, scorer, sampler)
        event = _start_event()
        vr, _, fps, duration = sampler.open_video("fake.mp4")
        refined = refiner.refine_event(event=event, vr=vr, fps=fps, duration=duration)

        assert isinstance(refined, RefinedEvent)
        assert refined is not event

    def test_source_is_original_event(self, config, sampler):
        scorer = _make_scorer()
        refiner = EventRefiner(config, scorer, sampler)
        event = _start_event()
        vr, _, fps, duration = sampler.open_video("fake.mp4")
        refined = refiner.refine_event(event=event, vr=vr, fps=fps, duration=duration)

        assert refined.source is event

    def test_original_event_unchanged(self, config, sampler):
        scorer = _make_scorer()
        refiner = EventRefiner(config, scorer, sampler)
        event = _start_event(time=5.0, score=0.8)
        vr, _, fps, duration = sampler.open_video("fake.mp4")
        refiner.refine_event(event=event, vr=vr, fps=fps, duration=duration)

        # Frozen dataclass — these would raise FrozenInstanceError if mutated
        assert event.time == pytest.approx(5.0)
        assert event.score == pytest.approx(0.8)

    def test_refinement_method_set(self, config, sampler):
        scorer = _make_scorer()
        refiner = EventRefiner(config, scorer, sampler)
        event = _start_event()
        vr, _, fps, duration = sampler.open_video("fake.mp4")
        refined = refiner.refine_event(event=event, vr=vr, fps=fps, duration=duration)

        assert refined.refinement_method != ""

    def test_refined_score_populated(self, config, sampler):
        scorer = _make_scorer(LABEL_START, 0.85)
        refiner = EventRefiner(config, scorer, sampler)
        event = _start_event()
        vr, _, fps, duration = sampler.open_video("fake.mp4")
        refined = refiner.refine_event(event=event, vr=vr, fps=fps, duration=duration)

        assert refined.refined_score >= 0.0
        assert refined.refined_frame >= 0

    def test_convenience_properties_delegate_to_source(self, config, sampler):
        scorer = _make_scorer()
        refiner = EventRefiner(config, scorer, sampler)
        event = _start_event()
        vr, _, fps, duration = sampler.open_video("fake.mp4")
        refined = refiner.refine_event(event=event, vr=vr, fps=fps, duration=duration)

        assert refined.label == LABEL_START
        assert refined.score == pytest.approx(0.8)


class TestChoiceRefine:
    def test_choice_event_uses_choice_strategy(self, config, sampler):
        scores = {l: 0.01 for l in LABEL_TEXTS}
        scores[LABEL_CHOICE] = 0.75
        scorer = ConstantScorer(scores)
        refiner = EventRefiner(config, scorer, sampler)
        event = _choice_event()
        vr, _, fps, duration = sampler.open_video("fake.mp4")
        refined = refiner.refine_event(event=event, vr=vr, fps=fps, duration=duration)

        assert isinstance(refined, RefinedEvent)
        assert refined.source is event
        assert "choice" in refined.refinement_method.lower()


class TestFilterEventsByFinalThreshold:
    def test_event_above_threshold_kept(self, config, sampler):
        scorer = _make_scorer()
        refiner = EventRefiner(config, scorer, sampler)
        event = _start_event()
        vr, _, fps, duration = sampler.open_video("fake.mp4")
        refined = refiner.refine_event(event=event, vr=vr, fps=fps, duration=duration)
        # Force a high enough score by using ConstantScorer with 0.85 > threshold 0.50
        result = refiner.filter_events_by_final_threshold([refined])
        assert len(result) == 1

    def test_event_below_threshold_removed(self, config, sampler):
        # Very low score scorer
        scores = {l: 0.01 for l in LABEL_TEXTS}
        scorer = ConstantScorer(scores)
        refiner = EventRefiner(config, scorer, sampler)
        event = _start_event()
        vr, _, fps, duration = sampler.open_video("fake.mp4")
        refined = refiner.refine_event(event=event, vr=vr, fps=fps, duration=duration)
        result = refiner.filter_events_by_final_threshold([refined])
        assert len(result) == 0
