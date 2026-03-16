"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from scripts.event_detector.config import DetectorConfig
from scripts.event_detector.labels import LABEL_TEXTS, LABEL_START, NONE_LABEL
from tests.event_detector.fakes import ConstantScorer, FakeVideoSampler


@pytest.fixture
def detector_config() -> DetectorConfig:
    return DetectorConfig()


@pytest.fixture
def constant_scorer() -> ConstantScorer:
    scores = {label: 0.01 for label in LABEL_TEXTS}
    scores[LABEL_START] = 0.85
    return ConstantScorer(scores)


@pytest.fixture
def fake_sampler() -> FakeVideoSampler:
    return FakeVideoSampler(num_frames=300, fps=30.0)
