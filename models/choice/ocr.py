from typing import List, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

from .config import LANG, USE_ANGLE_CLS, USE_GPU


def init_ocr() -> PaddleOCR:
    device = "gpu:0" if USE_GPU else "cpu"
    ocr = PaddleOCR(
        lang=LANG,
        device=device,
        use_textline_orientation=bool(USE_ANGLE_CLS),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        return_word_box=False,
    )
    return ocr


def run_paddleocr_lines_from_array(
    ocr: PaddleOCR, img_bgr: np.ndarray
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Returns list of (poly Nx2, text, score) for LINE-level results from an in-memory BGR image.
    Tries direct array inference; falls back to a temporary PNG if needed.
    """
    # Try direct array inference (some PaddleOCR builds support this)
    try:
        results = ocr.predict(img_bgr)
        if results:
            res = results[0]
            j = res.json
            r = j.get("res", j)

            polys = r.get("rec_polys", None)
            texts = r.get("rec_texts", None)
            scores = r.get("rec_scores", None)

            if polys is None:
                polys = r.get("dt_polys", [])
                texts = [""] * len(polys)
                scores = [0.0] * len(polys)

            if texts is None:
                texts = [""] * len(polys)
            if scores is None:
                scores = [0.0] * len(polys)

            n = min(len(polys), len(texts), len(scores))
            out = []
            for i in range(n):
                poly = np.asarray(polys[i], dtype=np.float32).reshape(-1, 2)
                out.append((poly, str(texts[i]), float(scores[i])))
            return out
    except Exception:
        print("Direct array OCR failed")


def run_ocr(ui_choice_groups):
    ocr = init_ocr()
    lines_dict = {}  # key -> list[(poly, text, score)]

    for item in ui_choice_groups:
        gid = item["gid"]
        masked_img = item["img"]

        if masked_img.ndim == 3 and masked_img.shape[2] == 3:
            img_bgr = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = masked_img

        lines = run_paddleocr_lines_from_array(ocr, img_bgr)

        key = f"group_{gid}"
        lines_dict[key] = lines

        if lines is None:
            print("Warning: No text detected in this crop.")
            continue

        print(f"\n{key}  (clf p={item['pred_prob']:.3f})")
        for i, (_, text, score) in enumerate(lines):
            if text.strip():
                print(f"  [{i:02d}] ({score:.2f}) {text}")

    return lines_dict
