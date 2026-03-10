import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow running this file directly: `uv run app_core/model_manager.py`
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from models.choice.choice_handler import ChoiceExtractor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from functools import lru_cache

import torch
from transformers import XCLIPModel, XCLIPProcessor

from models.choice.choice_extract import SelectionModel
from app_core.settings import CHOICE_SELECTION_MODEL_PATH, EVENT_DETECTOR_MODEL_DIR


class ModelManager:
    @staticmethod
    def get_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    @lru_cache(maxsize=1)
    def get_event_detector():
        model_dir = str(Path(EVENT_DETECTOR_MODEL_DIR))
        device = ModelManager.get_device()

        processor = XCLIPProcessor.from_pretrained(model_dir)
        model = XCLIPModel.from_pretrained(model_dir).to(device)
        model.eval()

        return {
            "processor": processor,
            "model": model,
            "device": device,
            "model_dir": model_dir,
            "num_model_frames": model.config.vision_config.num_frames,
        }

    @staticmethod
    @lru_cache(maxsize=1)
    def get_choice_extractor():
        sel = SelectionModel(weights_path=str(CHOICE_SELECTION_MODEL_PATH))
        ext = ChoiceExtractor(sel)
        return ext


if __name__ == "__main__":
    print("model loading...")
    extractor = ModelManager.get_choice_extractor()
    print("Model loaded successfully. processing...")
    res = extractor.process_frame("app_core/ex1.png")
    print(f"got: {res}")
