import os

import matplotlib.pyplot as plt

from .api import get_filtered_lines
from .classify import choice_classify
from .config import IMAGE_PATH, MAX_SHOW, PIPELINE_OUTPUT_DIR
from .detect import detect_ui
from .group_mask import group_masks
from .ocr import run_ocr
from .util import build_title_image_pairs


def run_pipeline(img_path):
    final_path = os.path.join(PIPELINE_OUTPUT_DIR, os.path.basename(img_path))
    os.makedirs(final_path, exist_ok=True)

    boxes_xyxy, w, h, img_rgb = detect_ui(img_path)
    grouped_masks = group_masks(boxes_xyxy, w, h, img_rgb, img_path)
    choices = choice_classify(grouped_masks, img_path)
    lines_dict = run_ocr(choices)
    lines_dict = get_filtered_lines(lines_dict)

    pairs = build_title_image_pairs(grouped_masks, lines_dict)
    print("titled pairs:", len(pairs))

    # --- Save ONLY titled pairs ---
    if not pairs:
        print("No titled detections found.")
    else:
        show_pairs = pairs[:MAX_SHOW]

        for i, d in enumerate(show_pairs):
            title = d["title"].upper().replace(" ", "_")
            filename = f"{i:03d}_{title}.png"
            path = os.path.join(final_path, filename)

            plt.imsave(path, d["crop"])

        print(f"Saved {len(show_pairs)} images to '{final_path}'")

    return pairs


if __name__ == "__main__":
    img_path = IMAGE_PATH
    run_pipeline(img_path)
