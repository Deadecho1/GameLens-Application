import os

import matplotlib.pyplot as plt
import numpy as np

from .config import MIN_MASK_AREA, OBJECT_GROUP_OUTPUT_DIR
from .util import (
    build_features_from_boxes,
    create_label_map_from_groups,
    group_boxes_dfs,
    overlay_colored_groups,
)


def group_masks(boxes_xyxy, w, h, img_rgb, img_path):
    if len(boxes_xyxy) == 0:
        print("Skipping grouping: no boxes from YOLO.")
        grouped_masks = []  # nothing to classify later
    else:
        # 1) build features
        boxes_xyxy_np, feats = build_features_from_boxes(boxes_xyxy, w, h)

        # 2) group
        size_ratio_thresh = 1.6
        align_thresh = 0.06
        strong_align_bonus = 1.0
        strong_align_thresh = 0.03

        groups = group_boxes_dfs(
            feats,
            size_ratio_thresh=size_ratio_thresh,
            align_thresh=align_thresh,
            strong_align_bonus=strong_align_bonus,
            strong_align_thresh=strong_align_thresh,
        )

        print(f"Stage 2: {len(boxes_xyxy)} boxes → {len(groups)} groups")
        for gid, inds in enumerate(groups, start=1):
            print(f"  Group {gid}: indices {inds}")

        # 3) label map
        label_map = create_label_map_from_groups(boxes_xyxy_np, groups, h, w)

        # 4) colored overlay
        overlay_img = overlay_colored_groups(img_rgb, label_map, alpha=0.5)

        # 5) per-group masked images in memory
        grouped_masks = []  # list of (group_id, masked_img_np)

        max_gid = int(label_map.max())
        fig_rows = max_gid if max_gid > 0 else 1
        fig, axes = plt.subplots(
            nrows=fig_rows, ncols=1, figsize=(6, 4 * fig_rows), squeeze=False
        )

        row_idx = 0
        for gid in range(1, max_gid + 1):
            mask = label_map == gid
            if not np.any(mask):
                continue

            masked_img = np.zeros_like(img_rgb)
            masked_img[mask] = img_rgb[mask]

            # Filter tiny masks
            if mask.sum() < MIN_MASK_AREA:
                continue

            grouped_masks.append((gid, masked_img))

            ax = axes[row_idx, 0]
            ax.imshow(masked_img)
            ax.set_title(f"Group {gid} mask")
            ax.axis("off")
            row_idx += 1

        # If we had fewer valid groups than max_gid, hide unused axes
        for r in range(row_idx, fig_rows):
            axes[r, 0].axis("off")

        plt.tight_layout()
        path = os.path.join(OBJECT_GROUP_OUTPUT_DIR, os.path.basename(img_path))
        plt.savefig(path)

    print(f"Total valid masks for Stage 3: {len(grouped_masks)}")
    return grouped_masks
