from PIL import Image

from app_core.cv2_compat import prepare_cv2_for_ultralytics

from .model import BossDetectionResult


class BossClassifier:
    """Loads a YOLO classification model once and classifies frames on demand."""

    def __init__(self, model_path: str) -> None:
        prepare_cv2_for_ultralytics()
        from ultralytics import YOLO

        self._model = YOLO(model_path)

    def classify_frame(self, image: Image.Image) -> BossDetectionResult:
        results = self._model.predict(source=image, save=False, verbose=False)
        result = results[0]

        probs = getattr(result, "probs", None)
        if probs is None or getattr(probs, "top1", None) is None:
            return BossDetectionResult(class_name="regular_gameplay", confidence=0.0)

        top_class_id = int(probs.top1)
        top_confidence = float(getattr(probs, "top1conf", 0.0))

        names = getattr(result, "names", None)
        if isinstance(names, dict):
            top_class_name = str(names.get(top_class_id, str(top_class_id)))
        else:
            top_class_name = str(top_class_id)

        return BossDetectionResult(class_name=top_class_name, confidence=top_confidence)
