"""Unit tests for JsonSerializer — no ML model required."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.event_detector.labels import LABEL_START, LABEL_END, LABEL_CHOICE
from scripts.event_detector.models import PeakEvent, RefinedEvent, RefinedRun
from scripts.event_detector.serializer import JsonSerializer


def _peak(label: str, time: float, score: float = 0.8) -> PeakEvent:
    return PeakEvent(label=label, time=time, score=score, start_time=time - 1.0, end_time=time + 1.0)


def _refined(peak: PeakEvent, refined_time: float | None = None) -> RefinedEvent:
    t = refined_time if refined_time is not None else peak.time
    return RefinedEvent(
        source=peak,
        refined_time=t,
        refined_frame=int(round(t * 30)),
        refined_score=peak.score,
        refined_window_start=t - 1.0,
        refined_window_end=t + 1.0,
        refinement_method="hillclimb",
    )


class TestEventToDict:
    def test_basic_fields_present(self):
        peak = _peak(LABEL_START, 5.0, 0.8)
        ev = _refined(peak)
        d = JsonSerializer.event_to_dict(ev)
        assert "time" in d
        assert "frame" in d
        assert "confidence" in d

    def test_uses_refined_time(self):
        peak = _peak(LABEL_START, 5.0)
        ev = _refined(peak, refined_time=5.25)
        d = JsonSerializer.event_to_dict(ev)
        assert d["time"] == pytest.approx(5.25)

    def test_retry_fields_omitted_when_empty(self):
        peak = _peak(LABEL_START, 5.0)
        ev = _refined(peak)
        d = JsonSerializer.event_to_dict(ev)
        assert "retry_frames" not in d
        assert "retry_times" not in d

    def test_retry_fields_present_when_set(self):
        peak = _peak(LABEL_CHOICE, 10.0)
        ev = RefinedEvent(
            source=peak,
            refined_time=10.0,
            refined_frame=300,
            refined_score=0.75,
            refined_window_start=9.0,
            refined_window_end=11.0,
            retry_frames=(295, 290),
            retry_times=(9.83, 9.67),
            refinement_method="choice-forward-exit-last-preferred",
        )
        d = JsonSerializer.event_to_dict(ev)
        assert d["retry_frames"] == [295, 290]
        assert d["retry_times"] == [9.83, 9.67]


class TestDecodedRunsToDict:
    def test_structure(self):
        ser = JsonSerializer()
        start = _refined(_peak(LABEL_START, 5.0))
        end = _refined(_peak(LABEL_END, 65.0))
        run = RefinedRun(start=start, end=end, choices=(), drops=())
        result = ser.decoded_runs_to_dict(
            video_path="/tmp/test.mp4",
            fps=30.0,
            duration=120.0,
            refined_runs=[run],
        )
        assert result["video_name"] == "test.mp4"
        assert result["fps"] == pytest.approx(30.0)
        assert result["duration_seconds"] == pytest.approx(120.0)
        assert len(result["runs"]) == 1
        assert result["runs"][0]["run_index"] == 1

    def test_choices_serialized(self):
        ser = JsonSerializer()
        start = _refined(_peak(LABEL_START, 5.0))
        end = _refined(_peak(LABEL_END, 65.0))
        choice = _refined(_peak(LABEL_CHOICE, 30.0))
        run = RefinedRun(start=start, end=end, choices=(choice,), drops=())
        result = ser.decoded_runs_to_dict("/tmp/v.mp4", 30.0, 120.0, [run])
        assert len(result["runs"][0]["choice_events"]) == 1


class TestSaveRunsJson:
    def test_writes_valid_json(self, tmp_path):
        ser = JsonSerializer()
        start = _refined(_peak(LABEL_START, 5.0))
        end = _refined(_peak(LABEL_END, 65.0))
        run = RefinedRun(start=start, end=end, choices=(), drops=())
        out = tmp_path / "video.json"
        ser.save_runs_json(
            video_path="/tmp/video.mp4",
            fps=30.0,
            duration=120.0,
            refined_runs=[run],
            json_out_path=str(out),
        )
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["video_name"] == "video.mp4"
        assert len(data["runs"]) == 1
