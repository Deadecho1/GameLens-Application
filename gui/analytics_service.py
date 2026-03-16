from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .models import ChoiceDetail, DashboardStats, RunDetails, RunSummary, VersionInfo


class AnalyticsService:
    def load_dashboard_stats(self, version: VersionInfo) -> DashboardStats:
        run_files = sorted(version.run_json_dir.glob("*.json"))
        durations: list[float] = []
        item_counter: Counter[str] = Counter()

        for path in run_files:
            data = self._load_json(path)
            runs = data if isinstance(data, list) else [data]
            for run in runs:
                duration = float(run.get("duration_seconds", 0.0) or 0.0)
                durations.append(duration)
                for choice in run.get("choices", []):
                    selected = choice.get("selected_option")
                    if selected:
                        item_counter[str(selected)] += 1

        if not durations:
            return DashboardStats(
                total_runs=0,
                average_run_duration_seconds=0.0,
                max_run_duration_seconds=0.0,
                min_run_duration_seconds=0.0,
                most_popular_item="—",
            )

        most_popular = item_counter.most_common(1)[0][0] if item_counter else "—"
        return DashboardStats(
            total_runs=len(durations),
            average_run_duration_seconds=sum(durations) / len(durations),
            max_run_duration_seconds=max(durations),
            min_run_duration_seconds=min(durations),
            most_popular_item=most_popular,
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

    def load_run_details(self, version: VersionInfo, run_summary: RunSummary) -> RunDetails:
        parts = run_summary.run_name.rsplit(" / Run ", 1)
        stem = parts[0]
        idx = int(parts[1]) - 1 if len(parts) == 2 else 0

        run_file = version.run_json_dir / f"{stem}.json"
        if not run_file.exists():
            return RunDetails(
                run_name=run_summary.run_name,
                duration_seconds=run_summary.duration_seconds,
                selected_items=run_summary.selected_items,
                choices=[],
            )

        data = self._load_json(run_file)
        runs = data if isinstance(data, list) else [data]
        if idx >= len(runs):
            idx = 0
        run = runs[idx]

        choices: list[ChoiceDetail] = []
        for choice in run.get("choices", []):
            options = choice.get("options", [])
            selected = str(choice.get("selected_option", "Unknown"))
            choices.append(
                ChoiceDetail(
                    options=[str(o) for o in options] if options else [],
                    selected=selected,
                )
            )

        return RunDetails(
            run_name=run_summary.run_name,
            duration_seconds=float(run.get("duration_seconds", 0.0) or 0.0),
            selected_items=run_summary.selected_items,
            choices=choices,
        )

    def _load_json(self, path: Path) -> dict | list:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
