"""Legacy settings module — delegates to AppConfig.

All new code should import from app_core.config.AppConfig directly.
These module-level names are kept for backwards compatibility.
"""
import os
from pathlib import Path

from app_core.config import AppConfig

_config = AppConfig.load()

PROJECT_ROOT = _config.project_root
MODELS_DIR = _config.models_dir
EVENT_DETECTOR_MODEL_DIR = _config.event_detector_model_dir

# fill this later when you add the real extractor assets/config
CHOICE_EXTRACTOR_DIR = MODELS_DIR / "choice_extractor"
CHOICE_SELECTION_MODEL_PATH = MODELS_DIR / "choice" / "models" / "selection_highlight_yolo" / "best.pt"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
