from __future__ import annotations

import json
from pathlib import Path

from .models import DashboardStats, RunSummary, VersionInfo


class AnalyticsService:
    def load_dashboard_stats(self, version: VersionInfo) -> DashboardStats:
        run_files = sorted(version.run_json_dir.glob("*.json"))
        durations: list[float] = []
        total_choices = 0

        for path in run_files:
            data = self._load_json(path)
            runs = data if isinstance(data, list) else [data]
            for run in runs:
                duration = float(run.get("duration_seconds", 0.0) or 0.0)
                durations.append(duration)
                total_choices += len(run.get("choices", []))

        if not durations:
            return DashboardStats(
                total_runs=0,
                average_run_duration_seconds=0.0,
                max_run_duration_seconds=0.0,
                min_run_duration_seconds=0.0,
                total_choices=0,
            )

        return DashboardStats(
            total_runs=len(durations),
            average_run_duration_seconds=sum(durations) / len(durations),
            max_run_duration_seconds=max(durations),
            min_run_duration_seconds=min(durations),
            total_choices=total_choices,
        )
    
    def load_run_summaries(self, version: VersionInfo) -> list[RunSummary]:
        summaries: list[RunSummary] = []
        for path in sorted(version.run_json_dir.glob("*.json")):
            data = self._load_json(path)
            runs = data if isinstance(data, list) else [data]
            for idx, run in enumerate(runs, start=1):
                choices = run.get("choices", [])
                item_names = [str(choice.get("selected_option", "Unknown")) for choice in choices]
                summaries.append(
                    RunSummary(
                        run_name=f"{path.stem} / Run {idx}",
                        duration_seconds=float(run.get("duration_seconds", 0.0) or 0.0),
                        selected_items=item_names,
                    )
                )
        return summaries

    def _load_json(self, path: Path) -> dict | list:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)