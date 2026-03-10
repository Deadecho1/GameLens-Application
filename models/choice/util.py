import json
import math

import cv2
import numpy as np

from .config import MAX_VDIST_PX, OVERLAP_THRESH, SCORE_OVERLAP_W, card_image_yolo_model


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def build_features_from_boxes(boxes_xyxy, img_w, img_h):
    """
    boxes_xyxy: (N,4) in pixel xyxy format.
    Returns:
      boxes_xyxy (unchanged),
      features: (N,4) [w_n, h_n, cx_n, cy_n] normalized to [0,1].
    """
    boxes_xyxy = np.array(boxes_xyxy, dtype=float)
    feats = []

    for x1, y1, x2, y2 in boxes_xyxy:
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        w_n = bw / img_w
        h_n = bh / img_h
        cx_n = cx / img_w
        cy_n = cy / img_h

        feats.append([w_n, h_n, cx_n, cy_n])

    return boxes_xyxy, np.array(feats, dtype=float)


def are_similar(
    feat_i,
    feat_j,
    size_ratio_thresh=1.6,  # base tolerance for width/height
    strong_align_bonus=0.8,  # extra slack when strongly aligned
    align_thresh=0.06,  # loose: "same column/row"
    strong_align_thresh=0.02,  # tight: "strongly aligned"
):
    """
    feat_*: [w_n, h_n, cx_n, cy_n] (normalized)
    """
    wi, hi, cxi, cyi = feat_i
    wj, hj, cxj, cyj = feat_j

    eps = 1e-6
    if wi < eps or hi < eps or wj < eps or hj < eps:
        return False

    # normalized edges
    x1_i = cxi - wi / 2.0
    x2_i = cxi + wi / 2.0
    y1_i = cyi - hi / 2.0
    y2_i = cyi + hi / 2.0

    x1_j = cxj - wj / 2.0
    x2_j = cxj + wj / 2.0
    y1_j = cyj - hj / 2.0
    y2_j = cyj + hj / 2.0

    # distances on X
    dx_c = abs(cxi - cxj)
    dx_l = abs(x1_i - x1_j)
    dx_r = abs(x2_i - x2_j)
    min_dx = min(dx_c, dx_l, dx_r)

    # distances on Y
    dy_c = abs(cyi - cyj)
    dy_t = abs(y1_i - y1_j)
    dy_b = abs(y2_i - y2_j)
    min_dy = min(dy_c, dy_t, dy_b)

    # alignment
    same_column = min_dx <= align_thresh
    same_row = min_dy <= align_thresh

    if not (same_column or same_row):
        return False

    strong_vertical = min_dx <= strong_align_thresh
    strong_horizontal = min_dy <= strong_align_thresh

    w_thresh = size_ratio_thresh
    h_thresh = size_ratio_thresh

    if strong_vertical:
        w_thresh = size_ratio_thresh + strong_align_bonus
    if strong_horizontal:
        h_thresh = size_ratio_thresh + strong_align_bonus

    w_ratio = max(wi, wj) / max(eps, min(wi, wj))
    h_ratio = max(hi, hj) / max(eps, min(hi, hj))

    size_ok = (w_ratio <= w_thresh) and (h_ratio <= h_thresh)
    return size_ok


def group_boxes_dfs(
    features,
    size_ratio_thresh=1.4,
    align_thresh=0.06,
    strong_align_bonus=0.6,
    strong_align_thresh=0.02,
):
    """
    features: (N, 4) [w_n, h_n, cx_n, cy_n]

    Returns:
      groups: list[list[int]]  (each list: indices of one UI group)
    """
    n = len(features)
    if n == 0:
        return []

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if are_similar(
                features[i],
                features[j],
                size_ratio_thresh=size_ratio_thresh,
                strong_align_bonus=strong_align_bonus,
                align_thresh=align_thresh,
                strong_align_thresh=strong_align_thresh,
            ):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        comp = []
        visited[i] = True

        while stack:
            k = stack.pop()
            comp.append(k)
            for nb in adj[k]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        groups.append(comp)

    return groups


def create_label_map_from_groups(boxes, groups, img_h, img_w):
    """
    boxes:  (N, 4) xyxy
    groups: list[list[int]]
    Returns:
      label_map: (H, W) int, 0=background, 1..K=group id
    """
    label_map = np.zeros((img_h, img_w), dtype=np.int32)

    for group_id, inds in enumerate(groups, start=1):
        for idx in inds:
            x1, y1, x2, y2 = boxes[idx]
            x1i = max(0, int(np.floor(x1)))
            y1i = max(0, int(np.floor(y1)))
            x2i = min(img_w, int(np.ceil(x2)))
            y2i = min(img_h, int(np.ceil(y2)))
            label_map[y1i:y2i, x1i:x2i] = group_id

    return label_map


def overlay_colored_groups(img, label_map, alpha=0.5):
    """
    img: RGB uint8 (H,W,3)
    label_map: (H,W) int, 0=background, 1..K=group ids

    Returns:
      RGB uint8 with each group colored differently.
    """
    img_f = img.astype(np.float32) / 255.0

    base_colors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.5, 0.0),
        (0.5, 0.0, 1.0),
        (0.5, 1.0, 0.0),
        (0.0, 0.5, 1.0),
        (1.0, 0.0, 0.5),
        (0.0, 1.0, 0.5),
    ]

    out = img_f.copy()
    max_group_id = int(label_map.max())

    for gid in range(1, max_group_id + 1):
        mask = label_map == gid
        if not np.any(mask):
            continue

        color = np.array(base_colors[(gid - 1) % len(base_colors)], dtype=np.float32)

        out[mask] = (1 - alpha) * img_f[mask] + alpha * color

    return (out * 255).astype(np.uint8)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure uint8 image for cv2/mpl."""
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def poly_to_xywh(poly: np.ndarray):
    xs = poly[:, 0]
    ys = poly[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return x0, y0, (x1 - x0), (y1 - y0)


def box_center(x, y, w, h):
    return (x + w / 2.0, y + h / 2.0)


def overlap_ratio_1d(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return 0.0 if union <= 0 else inter / union


def max_axis_overlap(a_xywh, b_xywh) -> float:
    ax, ay, aw, ah = a_xywh
    bx, by, bw, bh = b_xywh
    ox = overlap_ratio_1d(ax, ax + aw, bx, bx + bw)
    oy = overlap_ratio_1d(ay, ay + ah, by, by + bh)
    return max(ox, oy)


def vertical_distance_signed(text_xywh, det_xywh) -> float:
    tx, ty, tw, th = text_xywh
    dx, dy, dw, dh = det_xywh

    text_top, text_bot = ty, ty + th
    det_top, det_bot = dy, dy + dh

    if text_bot < det_top:
        return det_top - text_bot
    if det_bot < text_top:
        return text_top - det_bot
    return 0.0


def score_pair(text_xywh, det_xywh) -> float:
    overlap = max_axis_overlap(text_xywh, det_xywh)
    tcx, tcy = box_center(*text_xywh)
    dcx, dcy = box_center(*det_xywh)
    dist = math.hypot(tcx - dcx, tcy - dcy)
    return dist - SCORE_OVERLAP_W * overlap


def valid_pair(text_xywh, det_xywh) -> bool:
    overlap = max_axis_overlap(text_xywh, det_xywh)
    if overlap < OVERLAP_THRESH:
        return False
    vdist = vertical_distance_signed(text_xywh, det_xywh)
    return vdist <= MAX_VDIST_PX


def run_yolo_boxes(img_rgb: np.ndarray):
    """Return list of xywh boxes in image coordinates (Ultralytics-friendly)."""
    img_rgb = _to_uint8(img_rgb)

    try:
        res = card_image_yolo_model(img_rgb)
    except TypeError:
        res = card_image_yolo_model.predict(img_rgb)

    r0 = res[0] if isinstance(res, (list, tuple)) else res
    boxes = []

    if hasattr(r0, "boxes") and hasattr(r0.boxes, "xyxy"):
        xyxy = r0.boxes.xyxy
        xyxy = (
            xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.array(xyxy)
        )
        for x1, y1, x2, y2 in xyxy:
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            boxes.append((x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)))
        return boxes

    if hasattr(r0, "pred"):
        pred = r0.pred[0]
        pred = (
            pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.array(pred)
        )
        for row in pred:
            x1, y1, x2, y2 = row[:4]
            boxes.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
        return boxes

    raise RuntimeError("Could not extract YOLO boxes from model output.")


def assign_texts_to_dets(ocr_lines, det_boxes_xywh):
    """Assign each OCR line to the best matching detection (at most one det per line)."""
    texts = []
    for poly, text, ocr_score in ocr_lines:
        if not text or not text.strip():
            continue
        poly_np = np.asarray(poly, dtype=np.float32)
        texts.append(
            {
                "poly": poly_np,
                "xywh": poly_to_xywh(poly_np),
                "text": text.strip(),
                "ocr_score": float(ocr_score),
            }
        )

    assigned = {i: [] for i in range(len(det_boxes_xywh))}

    for t in texts:
        best_i = None
        best_s = float("inf")
        for i, dxywh in enumerate(det_boxes_xywh):
            if not valid_pair(t["xywh"], dxywh):
                continue
            s = score_pair(t["xywh"], dxywh)
            if s < best_s:
                best_s = s
                best_i = i
        if best_i is not None:
            assigned[best_i].append(t)

    return assigned


def concat_title_lines(lines):
    """Sort assigned OCR lines by (y, x) and join with spaces."""
    if not lines:
        return ""
    lines_sorted = sorted(lines, key=lambda d: (d["xywh"][1], d["xywh"][0]))
    return " ".join([d["text"] for d in lines_sorted]).strip()


def crop_xywh(img, xywh):
    h, w = img.shape[:2]
    x, y, bw, bh = xywh
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(w, int(round(x + bw)))
    y2 = min(h, int(round(y + bh)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def build_title_image_pairs(grouped_masks, lines_dict):
    pairs = []
    for gid, masked_img in grouped_masks:
        key = f"group_{gid}"
        ocr_lines = lines_dict.get(key, [])
        if not ocr_lines:
            continue

        img = _to_uint8(masked_img)
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img  # assume RGB; swap channels here if you feed BGR

        det_boxes = run_yolo_boxes(img_rgb)
        if not det_boxes:
            continue

        assigned = assign_texts_to_dets(ocr_lines, det_boxes)

        # IMPORTANT: title+image is per YOLO detection, not per group
        for det_i, dxywh in enumerate(det_boxes):
            title = concat_title_lines(assigned.get(det_i, []))
            if not title:
                continue  # skip no-title detections (also fixes your 'other groups' issue)
            crop = crop_xywh(img_rgb, dxywh)
            if crop is None:
                continue
            pairs.append(
                {
                    "gid": gid,
                    "det_i": det_i,
                    "title": title,
                    "crop": crop,
                    "det_xywh": dxywh,
                }
            )

    pairs.sort(key=lambda d: (d["gid"], d["det_i"]))
    return pairs
