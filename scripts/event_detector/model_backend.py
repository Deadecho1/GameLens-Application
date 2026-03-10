import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, List

from .labels import LABEL_TEXTS
from app_core.model_manager import ModelManager


class ModelBackend:
    def __init__(self):
        bundle = ModelManager.get_event_detector()
        self.device = bundle["device"]
        self.processor = bundle["processor"]
        self.model = bundle["model"]
        self.num_model_frames = bundle["num_model_frames"]
        self.text_inputs = self._prepare_text_inputs(LABEL_TEXTS)

    def _prepare_text_inputs(self, label_texts: List[str]) -> Dict:
        text_inputs = self.processor.tokenizer(
            label_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in text_inputs.items()}

    def score_clip(self, frames) -> Dict[str, float]:
        video_inputs = self.processor.image_processor(
            frames,
            return_tensors="pt",
        )

        pixel_values = video_inputs["pixel_values"].to(self.device)

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=self.text_inputs["input_ids"],
            attention_mask=self.text_inputs["attention_mask"],
        )

        logits = outputs.logits_per_video
        probs = logits.softmax(dim=-1)[0]

        return {
            LABEL_TEXTS[i]: float(probs[i].item())
            for i in range(len(LABEL_TEXTS))
        }