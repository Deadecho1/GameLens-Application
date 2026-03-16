from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional
    def load_dotenv(*args, **kwargs) -> None:  # type: ignore[misc]
        pass


@dataclass(frozen=True)
class AppConfig:
    # Paths
    project_root: Path
    models_dir: Path
    games_root: Path

    # Event detector
    event_detector_model_dir: Path

    # Choice extractor
    classifier_base_url: str
    classifier_timeout_seconds: float

    # GUI
    default_window_width: int
    default_window_height: int
    min_font_size: int
    max_font_size: int

    @classmethod
    def load(cls, env_file: Path | None = None) -> AppConfig:
        load_dotenv(env_file)

        project_root = Path(
            os.environ.get(
                "GAMELENS_PROJECT_ROOT",
                str(Path(__file__).resolve().parent.parent),
            )
        )
        models_dir = Path(os.environ.get("GAMELENS_MODELS_DIR", str(project_root / "models")))

        return cls(
            project_root=project_root,
            models_dir=models_dir,
            games_root=Path(
                os.environ.get("GAMELENS_GAMES_ROOT", str(project_root / "data" / "games"))
            ),
            event_detector_model_dir=Path(
                os.environ.get(
                    "GAMELENS_EVENT_DETECTOR_MODEL_DIR",
                    str(models_dir / "event_detector"),
                )
            ),
            classifier_base_url=os.environ.get("GAMELENS_CLASSIFIER_URL", "http://localhost:7761"),
            classifier_timeout_seconds=float(
                os.environ.get("GAMELENS_CLASSIFIER_TIMEOUT", "10.0")
            ),
            default_window_width=int(os.environ.get("GAMELENS_WINDOW_WIDTH", "1200")),
            default_window_height=int(os.environ.get("GAMELENS_WINDOW_HEIGHT", "800")),
            min_font_size=int(os.environ.get("GAMELENS_MIN_FONT_SIZE", "14")),
            max_font_size=int(os.environ.get("GAMELENS_MAX_FONT_SIZE", "20")),
        )
