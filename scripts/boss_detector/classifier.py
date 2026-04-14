from PIL import Image
from ultralytics import YOLO

from .model import BossDetectionResult


class BossClassifier:
    """Loads a YOLO classification model once and classifies frames on demand."""

    def __init__(self, model_path: str) -> None:
        self._model = YOLO(model_path)

    def classify_frame(self, image: Image.Image) -> BossDetectionResult:
        results = self._model.predict(source=image, save=False, verbose=False)
        result = results[0]
        top_class_id = result.probs.top1
        top_confidence = float(result.probs.top1conf)
        top_class_name = result.names[top_class_id]
        return BossDetectionResult(class_name=top_class_name, confidence=top_confidence)
