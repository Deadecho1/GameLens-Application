from __future__ import annotations

from pathlib import Path

APP_NAME = "GameLens"

# Path Constants
GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
OUTPUT_ROOT = PROJECT_ROOT / "data"
GAMES_ROOT = OUTPUT_ROOT / "games"

# GUI Constants
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800
MIN_FONT_SIZE = 14
MAX_FONT_SIZE = 20