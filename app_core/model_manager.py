import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from functools import lru_cache
from pathlib import Path

import torch
from transformers import XCLIPProcessor, XCLIPModel

from .settings import EVENT_DETECTOR_MODEL_DIR


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
        """
        TODO: Fill
        """