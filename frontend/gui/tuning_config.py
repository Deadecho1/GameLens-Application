from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TuningModelConfig:
    id: str
    label: str
    annotation_type: str   # "point" | "segment"
    train_script: str      # python -m <train_script> (final training step)
    user_category_ids: tuple[str, ...]  # point models: excludes auto-generated "none"


TUNING_MODEL_CONFIGS: list[TuningModelConfig] = [
    TuningModelConfig(
        id="event_detector",
        label="Event Detector",
        annotation_type="point",
        train_script="scripts.finetuner.cli",
        user_category_ids=("start", "end", "choice"),
    ),
    TuningModelConfig(
        id="boss_detector",
        label="Boss Detector",
        annotation_type="segment",
        train_script="scripts.finetuner.yolo_cli",
        user_category_ids=(),
    ),
]

TUNING_MODEL_CONFIG_BY_ID: dict[str, TuningModelConfig] = {
    c.id: c for c in TUNING_MODEL_CONFIGS
}
