import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import XCLIPModel, XCLIPProcessor

from app_core.settings import EVENT_DETECTOR_MODEL_DIR
from scripts.choice_extractor.extractor import ChoiceExtractor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        return ChoiceExtractor()


if __name__ == "__main__":
    print("loading choice extractor...")
    extractor = ModelManager.get_choice_extractor()

    with open("tests/output/clip18_frame.png", "rb") as f:
        img_bytes = f.read()

    res = extractor.extract_frame(img_bytes)
    print(f"got: {res}")
