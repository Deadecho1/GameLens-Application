from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    video_dir: Path
    event_json_dir: Path
    run_json_dir: Path
    only_events: bool
    only_export: bool
    verbose: bool