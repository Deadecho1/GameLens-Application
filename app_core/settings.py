from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

EVENT_DETECTOR_MODEL_DIR = MODELS_DIR / "event_detector"

# fill this later when you add the real extractor assets/config
CHOICE_EXTRACTOR_DIR = MODELS_DIR / "choice_extractor"
CHOICE_SELECTION_MODEL_PATH = MODELS_DIR / "choice" / "models" / "selection_highlight_yolo" / "best.pt"