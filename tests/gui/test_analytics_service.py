"""Unit tests for AnalyticsService using a fake JSON loader."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Union

import pytest

from gui.analytics_service import AnalyticsService
from gui.models import VersionInfo


def _make_version(tmp_path: Path) -> VersionInfo:
    run_dir = tmp_path / "run_json"
    run_dir.mkdir(parents=True, exist_ok=True)
    return VersionInfo(
        name="v1",
        root_dir=tmp_path,
        event_json_dir=tmp_path / "event_json",
        run_json_dir=run_dir,
    )


def _write_run_json(run_dir: Path, stem: str, runs: list) -> Path:
    path = run_dir / f"{stem}.json"
    path.write_text(json.dumps(runs))
    return path


class FakeJsonLoader:
    """Returns pre-set data regardless of path."""

    def __init__(self, data: Union[dict, list]) -> None:
        self._data = data

    def load(self, path: object) -> Union[dict, list]:
        return self._data


class TestLoadDashboardStats:
    def test_empty_dir_returns_zeros(self, tmp_path):
        version = _make_version(tmp_path)
        svc = AnalyticsService()
        stats = svc.load_dashboard_stats(version)
        assert stats.total_runs == 0
        assert stats.average_run_duration_seconds == pytest.approx(0.0)
        assert stats.most_popular_item == "—"

    def test_single_run_stats(self, tmp_path):
        version = _make_version(tmp_path)
        run_data = [{"duration_seconds": 120.0, "choices": [{"selected_option": "Sword"}]}]
        _write_run_json(version.run_json_dir, "clip1", run_data)
        svc = AnalyticsService()
        stats = svc.load_dashboard_stats(version)
        assert stats.total_runs == 1
        assert stats.average_run_duration_seconds == pytest.approx(120.0)
        assert stats.most_popular_item == "Sword"

    def test_multiple_runs_averages_correctly(self, tmp_path):
        version = _make_version(tmp_path)
        run_data = [
            {"duration_seconds": 60.0, "choices": []},
            {"duration_seconds": 120.0, "choices": []},
        ]
        _write_run_json(version.run_json_dir, "clip1", run_data)
        svc = AnalyticsService()
        stats = svc.load_dashboard_stats(version)
        assert stats.total_runs == 2
        assert stats.average_run_duration_seconds == pytest.approx(90.0)

    def test_most_popular_item_counted_correctly(self, tmp_path):
        version = _make_version(tmp_path)
        run_data = [
            {"duration_seconds": 60.0, "choices": [
                {"selected_option": "Shield"},
                {"selected_option": "Sword"},
                {"selected_option": "Shield"},
            ]},
        ]
        _write_run_json(version.run_json_dir, "clip1", run_data)
        svc = AnalyticsService()
        stats = svc.load_dashboard_stats(version)
        assert stats.most_popular_item == "Shield"


class TestLoadRunSummaries:
    def test_returns_one_summary_per_run(self, tmp_path):
        version = _make_version(tmp_path)
        run_data = [
            {"duration_seconds": 60.0, "choices": [{"selected_option": "A"}]},
            {"duration_seconds": 90.0, "choices": []},
        ]
        _write_run_json(version.run_json_dir, "clip1", run_data)
        svc = AnalyticsService()
        summaries = svc.load_run_summaries(version)
        assert len(summaries) == 2
        assert summaries[0].run_name == "clip1 / Run 1"
        assert summaries[1].run_name == "clip1 / Run 2"

    def test_summary_selected_items(self, tmp_path):
        version = _make_version(tmp_path)
        run_data = [{"duration_seconds": 60.0, "choices": [{"selected_option": "Bow"}]}]
        _write_run_json(version.run_json_dir, "clip1", run_data)
        svc = AnalyticsService()
        summaries = svc.load_run_summaries(version)
        assert summaries[0].selected_items == ["Bow"]


class TestLoadRunDetails:
    def test_full_choices_returned(self, tmp_path):
        version = _make_version(tmp_path)
        run_data = [
            {
                "duration_seconds": 60.0,
                "choices": [
                    {"selected_option": "Bow", "options": ["Sword", "Bow", "Shield"]},
                ],
            }
        ]
        _write_run_json(version.run_json_dir, "clip1", run_data)
        svc = AnalyticsService()
        summaries = svc.load_run_summaries(version)
        details = svc.load_run_details(version, summaries[0])
        assert len(details.choices) == 1
        assert details.choices[0].selected == "Bow"
        assert "Sword" in details.choices[0].options

    def test_missing_file_returns_summary_data(self, tmp_path):
        version = _make_version(tmp_path)
        from gui.models import RunSummary
        summary = RunSummary(run_name="missing / Run 1", duration_seconds=30.0, selected_items=[])
        svc = AnalyticsService()
        details = svc.load_run_details(version, summary)
        assert details.run_name == "missing / Run 1"
        assert details.choices == []


class TestCaching:
    def test_second_call_uses_cache(self, tmp_path):
        version = _make_version(tmp_path)
        run_data = [{"duration_seconds": 60.0, "choices": []}]
        _write_run_json(version.run_json_dir, "clip1", run_data)
        svc = AnalyticsService()
        stats1 = svc.load_dashboard_stats(version)
        stats2 = svc.load_dashboard_stats(version)
        assert stats1 is stats2  # Same object from cache

    def test_cache_invalidated_after_new_file(self, tmp_path):
        version = _make_version(tmp_path)
        run_data = [{"duration_seconds": 60.0, "choices": []}]
        _write_run_json(version.run_json_dir, "clip1", run_data)
        svc = AnalyticsService()
        stats1 = svc.load_dashboard_stats(version)
        assert stats1.total_runs == 1

        # Add another file (sleep briefly to ensure mtime differs)
        time.sleep(0.01)
        _write_run_json(version.run_json_dir, "clip2", run_data)
        stats2 = svc.load_dashboard_stats(version)
        assert stats2.total_runs == 2
