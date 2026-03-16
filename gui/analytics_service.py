from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import ChoiceDetail, DashboardStats, RunDetails, RunSummary, VersionInfo


class StdJsonLoader:
    """Standard filesystem JSON loader."""

    def load(self, path: Path) -> dict | list:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


class AnalyticsService:
    def __init__(self, json_loader=None) -> None:
        self._loader = json_loader or StdJsonLoader()
        self._stats_cache: Dict[Path, DashboardStats] = {}
        self._summaries_cache: Dict[Path, List[RunSummary]] = {}
        self._fingerprints: Dict[Path, Tuple] = {}

    def _fingerprint(self, run_json_dir: Path) -> Tuple:
        files = sorted(run_json_dir.glob("*.json"))
        return tuple((p, p.stat().st_mtime) for p in files)

    def _is_cache_valid(self, run_json_dir: Path) -> bool:
        return self._fingerprints.get(run_json_dir) == self._fingerprint(run_json_dir)

    def _update_fingerprint(self, run_json_dir: Path) -> List[Path]:
        files = sorted(run_json_dir.glob("*.json"))
        self._fingerprints[run_json_dir] = tuple((p, p.stat().st_mtime) for p in files)
        return files

    def load_dashboard_stats(self, version: VersionInfo) -> DashboardStats:
        key = version.run_json_dir
        if key in self._stats_cache and self._is_cache_valid(key):
            return self._stats_cache[key]

        run_files = self._update_fingerprint(key)
        durations: list[float] = []
        item_counter: Counter[str] = Counter()

        for path in run_files:
            data = self._loader.load(path)
            runs = data if isinstance(data, list) else [data]
            for run in runs:
                duration = float(run.get("duration_seconds") or run.get("duration") or 0.0)
                durations.append(duration)
                for choice in run.get("choices", []):
                    selected = choice.get("selected_option") or choice.get("selected_choice")
                    if selected:
                        item_counter[str(selected)] += 1

        if not durations:
            result = DashboardStats(
                total_runs=0,
                average_run_duration_seconds=0.0,
                max_run_duration_seconds=0.0,
                min_run_duration_seconds=0.0,
                most_popular_item="—",
            )
        else:
            most_popular = item_counter.most_common(1)[0][0] if item_counter else "—"
            result = DashboardStats(
                total_runs=len(durations),
                average_run_duration_seconds=sum(durations) / len(durations),
                max_run_duration_seconds=max(durations),
                min_run_duration_seconds=min(durations),
                most_popular_item=most_popular,
            )

        self._stats_cache[key] = result
        return result

    def load_run_summaries(self, version: VersionInfo) -> list[RunSummary]:
        key = version.run_json_dir
        if key in self._summaries_cache and self._is_cache_valid(key):
            return self._summaries_cache[key]

        run_files = self._update_fingerprint(key)
        summaries: list[RunSummary] = []
        for path in run_files:
            data = self._loader.load(path)
            runs = data if isinstance(data, list) else [data]
            for idx, run in enumerate(runs, start=1):
                choices = run.get("choices", [])
                item_names = [str(choice.get("selected_option") or choice.get("selected_choice") or "Unknown") for choice in choices]
                summaries.append(
                    RunSummary(
                        run_name=f"{path.stem} / Run {idx}",
                        duration_seconds=float(run.get("duration_seconds") or run.get("duration") or 0.0),
                        selected_items=item_names,
                    )
                )

        self._summaries_cache[key] = summaries
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

        data = self._loader.load(run_file)
        runs = data if isinstance(data, list) else [data]
        if idx >= len(runs):
            idx = 0
        run = runs[idx]

        choices: list[ChoiceDetail] = []
        for choice in run.get("choices", []):
            options = choice.get("options", [])
            selected = str(choice.get("selected_option") or choice.get("selected_choice") or "Unknown")
            choices.append(
                ChoiceDetail(
                    options=[str(o) for o in options] if options else [],
                    selected=selected,
                )
            )

        return RunDetails(
            run_name=run_summary.run_name,
            duration_seconds=float(run.get("duration_seconds") or run.get("duration") or 0.0),
            selected_items=run_summary.selected_items,
            choices=choices,
        )
