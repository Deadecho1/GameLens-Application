"""
yolo_trainer.py — YOLO classification fine-tuning for the boss detector.

Adapts the base boss YOLO model to a new game using a small labelled
frame dataset (boss/ and regular_gameplay/ subfolders).
Uses Ultralytics model.train() API — no custom training loop needed.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app_core.logging import get_logger

logger = get_logger(__name__)

EXPECTED_CLASSES = {"boss", "regular_gameplay"}


def run_finetuning(
    clips_dir: Path,
    base_model_path: Path,
    output_dir: Path,
    epochs: int = 50,
    imgsz: int = 224,
    seed: int = 42,
) -> None:
    """
    Fine-tune YOLO boss classifier on frames in clips_dir and save best.pt to output_dir.

    clips_dir must contain subfolders: boss/ and/or regular_gameplay/
    output_dir will contain best.pt compatible with BossClassifier.
    """
    from ultralytics import YOLO

    found_classes = {d.name for d in clips_dir.iterdir() if d.is_dir()}
    if not found_classes.intersection(EXPECTED_CLASSES):
        raise ValueError(
            f"No expected class folders in {clips_dir}. "
            f"Expected {EXPECTED_CLASSES}, found: {found_classes}"
        )

    for cls in sorted(found_classes.intersection(EXPECTED_CLASSES)):
        count = len(list((clips_dir / cls).glob("*.jpg")))
        logger.info("%s frames: %d", cls, count)

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    logger.info("Loading base model from %s", base_model_path)
    model = YOLO(str(base_model_path))

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting YOLO fine-tuning: epochs=%d imgsz=%d", epochs, imgsz)
    model.train(
        data=str(clips_dir),
        epochs=epochs,
        imgsz=imgsz,
        seed=seed,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=True,
    )

    # Copy best.pt to output root for BossClassifier compatibility
    weights_best = output_dir / "weights" / "best.pt"
    final_path = output_dir / "best.pt"
    if weights_best.exists():
        shutil.copy2(str(weights_best), str(final_path))
        logger.info("Best model saved to %s", final_path)
    else:
        logger.warning("best.pt not found at expected path: %s", weights_best)

    logger.info("Done.")
