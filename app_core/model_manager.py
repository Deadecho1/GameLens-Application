from pathlib import Path

import torch
from transformers import XCLIPModel, XCLIPProcessor

from scripts.choice_extractor.extractor import ChoiceExtractor, ChoiceExtractorConfig


class ModelManager:
    @staticmethod
    def get_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def load_event_detector(model_dir: Path) -> dict:
        """Load and return the XCLIP event detector bundle.

        Callers are responsible for caching the result if desired.
        """
        device = ModelManager.get_device()
        model_path = str(model_dir)

        processor = XCLIPProcessor.from_pretrained(model_path)
        model = XCLIPModel.from_pretrained(model_path).to(device)
        model.eval()

        return {
            "processor": processor,
            "model": model,
            "device": device,
            "model_dir": model_path,
            "num_model_frames": model.config.vision_config.num_frames,
        }

    @staticmethod
    def load_choice_extractor(base_url: str = "http://localhost:7761", timeout: float = 10.0) -> ChoiceExtractor:
        """Create and return a ChoiceExtractor instance."""
        return ChoiceExtractor(config=ChoiceExtractorConfig(base_url=base_url, timeout_seconds=timeout))


if __name__ == "__main__":
    print("loading choice extractor...")
    extractor = ModelManager.load_choice_extractor()

    with open("tests/output/clip18_frame.png", "rb") as f:
        img_bytes = f.read()

    res = extractor.extract_frame(img_bytes)
    print(f"got: {res}")
