from dataclasses import dataclass, field

from .labels import LABEL_START, LABEL_END, LABEL_DROP, LABEL_CHOICE


@dataclass(frozen=True)
class DetectorConfig:
    window_seconds: float = 2.0
    stride_seconds: float = 2.0

    raw_candidate_threshold: float = 0.20
    peak_neighborhood: int = 1

    final_event_score_thresholds: dict = field(default_factory=lambda: {
        LABEL_START: 0.50,
        LABEL_END: 0.50,
        LABEL_CHOICE: 0.60,
        LABEL_DROP: 0.60,
    })

    merge_gap_seconds: dict = field(default_factory=lambda: {
        LABEL_START: 6.0,
        LABEL_END: 6.0,
        LABEL_DROP: 4.0,
        LABEL_CHOICE: 4.0,
    })

    merge_boundary_pad_seconds: float = 1.0

    min_run_seconds: float = 8.0
    short_run_penalty_weight: float = 0.90
    inside_event_reward_weight: float = 0.55
    outside_event_penalty_weight: float = 0.20
    inner_boundary_penalty_weight: float = 0.35

    hillclimb_window_seconds: float = 2.0
    refine_grid_step_seconds: float = 0.25
    hillclimb_step_seconds: float = 0.25
    hillclimb_max_iters: int = 30