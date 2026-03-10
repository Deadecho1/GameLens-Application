# choice_extract.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO

from .pipeline import run_pipeline  # uses our existing poc pipeline
# run_pipeline returns list of dicts with keys like: title, det_xywh, crop, gid, det_i :contentReference[oaicite:3]{index=3}
# det_xywh comes from card-image yolo on masked_img, which is same size as original, so coords are frame coords :contentReference[oaicite:4]{index=4}


# ----------------------------
# Helpers
# ----------------------------
def xywh_to_xyxy(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return (x, y, x + w, y + h)

def xyxy_to_xywh(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))

def iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union

def contains_xywh(outer: Tuple[float,float,float,float], inner: Tuple[float,float,float,float], slack: float = 0.0) -> bool:
    ox1, oy1, ox2, oy2 = xywh_to_xyxy(outer)
    ix1, iy1, ix2, iy2 = xywh_to_xyxy(inner)
    return (ix1 >= ox1 - slack and iy1 >= oy1 - slack and ix2 <= ox2 + slack and iy2 <= oy2 + slack)


# ----------------------------
# Selection model wrapper
# ----------------------------
@dataclass
class HighlightDet:
    bbox_xywh: Tuple[float, float, float, float]
    conf: float
    cls_name: str

class SelectionModel:
    def __init__(self, weights_path: str, highlight_class_name: Optional[str] = None, conf: float = 0.75):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.highlight_class_name = highlight_class_name  # if None, take top-1 box regardless of class

    def detect_highlight(self, img_path: str) -> Optional[HighlightDet]:
        res = self.model(img_path, conf=self.conf)
        r0 = res[0]

        if r0.boxes is None or len(r0.boxes) == 0:
            return None

        boxes = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy()

        # id->name mapping if exists
        names = getattr(r0, "names", None) or getattr(self.model, "names", None) or {}
        def cls_name(ci: int) -> str:
            return names.get(int(ci), str(int(ci)))

        # filter by class name if requested
        candidates = []
        for xyxy, c, ci in zip(boxes, confs, clss):
            name = cls_name(int(ci))
            if self.highlight_class_name is None or name == self.highlight_class_name:
                x1, y1, x2, y2 = map(float, xyxy)
                candidates.append((float(c), name, (x1, y1, x2, y2)))

        if not candidates:
            return None

        # choose highest confidence
        c, name, xyxy = max(candidates, key=lambda t: t[0])
        return HighlightDet(bbox_xywh=xyxy_to_xywh(xyxy), conf=c, cls_name=name)


# ----------------------------
# Policy + extraction
# ----------------------------
@dataclass
class ChoiceResult:
    frame_path: str
    chosen_title: str
    chosen_bbox_xywh: Tuple[float, float, float, float]
    highlight_bbox_xywh: Tuple[float, float, float, float]
    options: List[Dict[str, Any]]  # raw pairs from run_pipeline


def extract_choice_from_interval(
    frame_paths: List[str],
    selection_weights: str,
    *,
    highlight_class_name: Optional[str] = None,   # set if your model has a specific class name
    min_options: int = 2,
    max_frames_to_try: int = 12,
    selection_conf: float = 0.75,
    match_iou_thresh: float = 0.10,
) -> Optional[ChoiceResult]:
    """
    frame_paths: ordered in time (oldest -> newest).
    Policy: scan backwards to find the *last* frame with highlight detection, try that first.
    Fallback: if options extraction or matching fails, try earlier frames.
    """

    sel = SelectionModel(selection_weights, highlight_class_name=highlight_class_name, conf=selection_conf)

    # 1) POLICY: build candidate list from newest->oldest, preferring frames where highlight exists
    newest_first = list(reversed(frame_paths))
    candidates: List[Tuple[str, Optional[HighlightDet]]] = []
    for p in newest_first:
        h = sel.detect_highlight(p)
        candidates.append((p, h))

    # prioritize: highlight exists first, then by recency
    candidates.sort(key=lambda t: (t[1] is None, ), reverse=False)
    # after sort: highlight frames first, still roughly recent; we then iterate with a cap
    tried = 0

    for frame_path, hdet in candidates:
        if tried >= max_frames_to_try:
            break
        tried += 1

        # If no highlight at all, this frame is weak; still allow it only after highlight frames get exhausted.
        if hdet is None:
            continue

        # 2) Extract options
        try:
            pairs = run_pipeline(frame_path)
        except Exception:
            continue

        if not pairs or len(pairs) < min_options:
            continue

        # 3) Match highlight bbox to one of the extracted option bboxes
        hb = hdet.bbox_xywh

        best = None
        best_score = -1.0

        for opt in pairs:
            ob = tuple(map(float, opt.get("det_xywh", (0, 0, 0, 0))))
            if ob[2] <= 0 or ob[3] <= 0:
                continue

            # score by IoU; also accept containment with slack
            score = iou_xywh(hb, ob)
            if score < 1e-6 and contains_xywh(ob, hb, slack=10.0):
                score = 0.5  # treat as decent match if highlight is inside option

            if score > best_score:
                best_score = score
                best = opt

        if best is None:
            continue

        if best_score < match_iou_thresh:
            # try another frame
            continue

        return ChoiceResult(
            frame_path=frame_path,
            chosen_title=str(best["title"]),
            chosen_bbox_xywh=tuple(map(float, best["det_xywh"])),
            highlight_bbox_xywh=hb,
            options=pairs,
        )

    # If we got here: no highlight-based match worked.
    # Optional extra fallback: try frames without highlight by choosing "best looking" options frame.
    # For now, return None.
    return None


if __name__ == "__main__":
    # Example usage:
    frames = [
        # fill with your interval frames, oldest->newest
        # r"C:\...\frame_0001.png",
    ]
    res = extract_choice_from_interval(
        frames,
        selection_weights="models/selection_highlight_yolo/best.pt",
        highlight_class_name="selection",   
    )
    if res is None:
        print("No choice extracted.")
    else:
        print("Frame:", res.frame_path)
        print("Chosen:", res.chosen_title)
        print("Chosen bbox:", res.chosen_bbox_xywh)
        print("Highlight bbox:", res.highlight_bbox_xywh)
        print("Options:", [o["title"] for o in res.options])