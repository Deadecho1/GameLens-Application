from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VersionInfo:
    name: str
    root_dir: Path
    event_json_dir: Path
    run_json_dir: Path


@dataclass(frozen=True)
class PipelineConfig:
    video_dir: Path
    event_json_dir: Path
    run_json_dir: Path
    only_events: bool
    only_export: bool
    verbose: bool


@dataclass(frozen=True)
class DashboardStats:
    total_runs: int
    average_run_duration_seconds: float
    max_run_duration_seconds: float
    min_run_duration_seconds: float
    total_choices: int


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    duration_seconds: float
    selected_items: list[str]