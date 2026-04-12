import argparse

from model import BossDetectionResult
from PIL import Image
from ultralytics import YOLO


def classify_image(model_path: str, image_data: Image.Image) -> BossDetectionResult:
    """
    Loads a YOLO classification model, runs inference on a PIL image,
    and returns the top prediction.
    """
    model = YOLO(model_path)

    results = model.predict(source=image_data, save=False)
    result = results[0]

    top_class_id = result.probs.top1
    top_confidence = float(result.probs.top1conf)
    top_class_name = result.names[top_class_id]

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
