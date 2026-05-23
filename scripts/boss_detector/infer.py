import argparse

try:
    from .model import BossDetectionResult
except ImportError:  # fallback when run as __main__ script
    from model import BossDetectionResult  # type: ignore[no-redef]
from PIL import Image

from app_core.cv2_compat import prepare_cv2_for_ultralytics


def classify_image(model_path: str, image_data: Image.Image) -> BossDetectionResult:
    """
    Loads a YOLO classification model, runs inference on a PIL image,
    and returns the top prediction.
    """
    prepare_cv2_for_ultralytics()
    from ultralytics import YOLO

    model = YOLO(model_path)

    results = model.predict(source=image_data, save=False)
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

    res = BossDetectionResult(class_name=top_class_name, confidence=top_confidence)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    loaded_image = Image.open(args.image)
    prediction = classify_image(model_path=args.model, image_data=loaded_image)
    print(prediction)
