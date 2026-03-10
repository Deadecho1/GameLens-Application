import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from ultralytics import YOLO

# -------------------------
# Base paths (fixes "relative paths" issue)
# -------------------------
# Directory that contains THIS config.py (…/event_extraction/workers/choice)
_THIS_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _THIS_DIR / "models"

# -------------------------
# Output dirs (keep names the same)
# -------------------------
PIPELINE_OUTPUT_DIR = "model_results/pipeline"
OCR_OUTPUT_DIR = "model_results/ocr"
CHOICE_CLASSIFIER_OUTPUT_DIR = "model_results/ui_choice_classifier"
OBJECT_GROUP_OUTPUT_DIR = "model_results/ui_object_merge"
DETECTION_OUTPUT_DIR = "model_results/detect"
DEBUG_OUTPUT_DIR = "model_results/debug"

os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)
os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHOICE_CLASSIFIER_OUTPUT_DIR, exist_ok=True)
os.makedirs(OBJECT_GROUP_OUTPUT_DIR, exist_ok=True)
os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Model paths (keep names the same, but make them correct)
# -------------------------
YOLO_WEIGHTS = str((_MODELS_DIR / "ui_detector_yolov12" / "weights" / "best.pt").resolve())

IMAGE_PATH = "results/ui_detector/input_examples/img3.jpg"

CONVNEXT_DIR = str((_MODELS_DIR / "ui_choice_classifier_convnextv2").resolve())

# Minimum area for a mask to be considered non-empty (to filter noise)
MIN_MASK_AREA = 50  # pixels

device = "cuda" if torch.cuda.is_available() else "cpu"

# paddleOCR settings
USE_GPU: bool = True if device == "cuda" else False
LANG: str = "en"
USE_ANGLE_CLS: bool = True

print("Using device:", device)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Image model Tunables ---
OVERLAP_THRESH = 0.05  # minimum overlap ratio to consider a text<->det pairing
MAX_VDIST_PX = 60  # max allowed vertical separation (px) between text box and det box
SCORE_OVERLAP_W = 300.0  # overlap weight in score = dist - W*overlap

# Visualization
MAX_SHOW = 200

card_image_yolo_model = YOLO(
    str((_MODELS_DIR / "ui_choice_image_detection" / "weights" / "best.pt").resolve())
)