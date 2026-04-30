"""GUI config — delegates to AppConfig.

All new code should use AppConfig directly. These module-level names are kept
for backwards compatibility with existing gui imports.
"""
from __future__ import annotations

from pathlib import Path

from app_core.config import AppConfig

_config = AppConfig.load()

APP_NAME = "GameLens"

# Path constants
GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _config.project_root
OUTPUT_ROOT = _config.project_root / "data"
GAMES_ROOT = _config.games_root

# GUI constants
DEFAULT_WINDOW_WIDTH = _config.default_window_width
DEFAULT_WINDOW_HEIGHT = _config.default_window_height
MIN_FONT_SIZE = _config.min_font_size
MAX_FONT_SIZE = _config.max_font_size
