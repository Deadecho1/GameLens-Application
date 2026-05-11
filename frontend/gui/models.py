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
    game_name: str = ""
    version_name: str = ""
    user_id: str = "1"
    collector_url: str = "http://localhost:8000"
    model_dir: Path | None = None  # if set, overrides GAMELENS_EVENT_DETECTOR_MODEL_DIR
