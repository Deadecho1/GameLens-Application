from __future__ import annotations

from pathlib import Path

APP_NAME = "GameLens"

# Path Constants
GUI_DIR = Path(__file__).resolve().parent 
PROJECT_ROOT = GUI_DIR.parent
DEFAULT_EVENT_JSON_DIR = PROJECT_ROOT / "data" / "event_json"
DEFAULT_RUN_JSON_DIR = PROJECT_ROOT / "data" / "run_json"

# GUI Constants
DEFAULT_WINDOW_WIDTH = 1000
DEFAULT_WINDOW_HEIGHT = 720
MIN_FONT_SIZE = 12
MAX_FONT_SIZE = 18