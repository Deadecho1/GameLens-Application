from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TuningModelConfig:
    id: str
    label: str
    train_script: str  # python -m <train_script>
    user_category_ids: tuple[str, ...]  # excludes 'none' (auto-generated)


TUNING_MODEL_CONFIGS: list[TuningModelConfig] = [
    TuningModelConfig(
        id="event_detector",
        label="Event Detector",
        train_script="scripts.finetuner.cli",
        user_category_ids=("start", "end", "choice"),
    ),
]

TUNING_MODEL_CONFIG_BY_ID: dict[str, TuningModelConfig] = {
    c.id: c for c in TUNING_MODEL_CONFIGS
}
