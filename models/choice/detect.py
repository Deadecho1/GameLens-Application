import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from .config import DETECTION_OUTPUT_DIR, YOLO_WEIGHTS

# Load YOLO model
yolo_model = YOLO(YOLO_WEIGHTS)


def detect_ui(img_path: str):
    # Run inference
    yolo_results = yolo_model(img_path, conf=0.25)
    result = yolo_results[0]

    # Read image as RGB (numpy)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Extract boxes in xyxy format
    if result.boxes is None or len(result.boxes) == 0:
        print("No detections from YOLO.")
        boxes_xyxy = np.zeros((0, 4), dtype=float)
    else:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N,4)

    print(f"YOLO detections: {len(boxes_xyxy)} boxes")

    # Visualize YOLO detections
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb)
    for x1, y1, x2, y2 in boxes_xyxy:
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2
        )
        ax.add_patch(rect)
    ax.set_title(f"Stage 1: YOLO detections ({len(boxes_xyxy)} boxes)")
    ax.axis("off")
    path = os.path.join(DETECTION_OUTPUT_DIR, os.path.basename(img_path))
    plt.savefig(path)

    return boxes_xyxy, w, h, img_rgb
