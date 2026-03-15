from dataclasses import dataclass, field

from .labels import LABEL_START, LABEL_END, LABEL_DROP, LABEL_CHOICE


@dataclass(frozen=True)
class DetectorConfig:
    # Length of each clip fed into the classifier (seconds).
    # This should roughly match the clip length used during training.
    window_seconds: float = 2.0

    # Time shift between consecutive clips when scanning the video.
    # Lower values = more overlap = higher detection accuracy but slower.
    stride_seconds: float = 2.0

    # Minimum raw classifier confidence required to keep a candidate event.
    # Events below this score are discarded before peak detection.
    raw_candidate_threshold: float = 0.20

    # Number of neighboring samples required when identifying local peaks.
    # Helps suppress noisy spikes from the classifier.
    peak_neighborhood: int = 1

    # Final minimum score required for an event after merging/refinement.
    # Different event types can require different confidence levels.
    final_event_score_thresholds: dict = field(default_factory=lambda: {
        LABEL_START: 0.50,
        LABEL_END: 0.50,
        LABEL_CHOICE: 0.60,
        LABEL_DROP: 0.60,
    })

    # Maximum allowed time gap between events to consider them part of the
    # same merged event region.
    # Helps combine multiple classifier peaks for the same real event.
    merge_gap_seconds: dict = field(default_factory=lambda: {
        LABEL_START: 6.0,
        LABEL_END: 6.0,
        LABEL_DROP: 4.0,
        LABEL_CHOICE: 4.0,
    })

    # Extra padding added to merged event boundaries.
    # Ensures refinement has room to search around the detected peak.
    merge_boundary_pad_seconds: float = 1.0

    # Minimum allowed run duration in seconds.
    # Runs shorter than this receive penalties during run decoding.
    min_run_seconds: float = 8.0

    # Penalty weight applied to runs shorter than min_run_seconds.
    short_run_penalty_weight: float = 0.90

    # Reward weight for events that appear inside a valid run interval.
    inside_event_reward_weight: float = 0.55

    # Penalty applied to events that appear outside run boundaries.
    outside_event_penalty_weight: float = 0.20

    # Penalty applied when events occur close to run boundaries.
    # Helps avoid placing events exactly at run edges.
    inner_boundary_penalty_weight: float = 0.35

    # Search window size (seconds) for hill-climbing refinement of events
    # like start/end/drop where we want to maximize classifier confidence.
    hillclimb_window_seconds: float = 2.0

    # Grid spacing (seconds) used for the initial refinement search.
    # Smaller values = more precise but slower.
    refine_grid_step_seconds: float = 0.25

    # Step size used by the hill-climbing optimizer when shifting the
    # candidate event position.
    hillclimb_step_seconds: float = 0.25

    # Maximum number of hill-climb iterations allowed during refinement.
    hillclimb_max_iters: int = 30

    # ---- Choice event refinement parameters ----
    # Choice events use a forward-first refinement strategy:
    # start near the coarse peak, search forward for the first exit-like
    # transition out of a valid choice screen, and only fall back to a
    # small lookback window if forward search fails.

    # How far before the coarse peak the choice refiner is allowed to look
    # when the coarse peak lands slightly after the real selection moment.
    # Keep this small so the refiner does not drift all the way back to the
    # gameplay -> choice-screen entry transition.
    choice_peak_lookback_seconds: float = 0.40

    # Extra time to search after the merged choice region ends.
    # This helps when the true exit transition starts a bit after the merged
    # region boundary.
    choice_refine_region_pad_after_seconds: float = 0.80

    # Number of frames used when smoothing frame-difference scores.
    # Reduces noise when detecting visual transitions.
    choice_diff_smooth_window: int = 5

    # Frame sampling step when computing frame differences.
    # 1 = every frame, 2 = every second frame, etc.
    # Larger values improve speed but may miss fast transitions.
    choice_frame_step: int = 1

    # Minimum normalized transition strength required before a forward diff
    # spike is treated as a plausible exit transition. This is applied to an
    # adaptive threshold derived from the local diff statistics.
    choice_exit_min_strength_ratio: float = 0.60

    # Number of consecutive scored candidate frames that must show weakened
    # choice confidence after a transition before it is accepted as an exit.
    # Larger values are more conservative but may miss very short exits.
    choice_exit_confirm_frames: int = 2

    # Minimum required drop in choice score after a candidate transition for
    # it to count as a real exit from the choice screen plateau.
    choice_exit_score_drop: float = 0.08

    # Frames to retry if the primary extracted frame fails downstream.
    # These are offsets backward from the selected frame.
    # Example: (0,1,2) means try frame, frame-1, frame-2.
    choice_retry_lookback_frames: tuple[int, ...] = (0, 5, 10, 20, 30, 60)

    # Time spacing between candidate frames evaluated by the classifier
    # when validating choice screens.
    # Smaller values = denser validation but slower runtime.
    choice_clip_score_grid_step_seconds: float = 0.10

    # Minimum classifier score required for a frame to be considered
    # a valid choice screen candidate.
    choice_min_clip_score: float = 0.18

    # Preferred score indicating a strong choice-screen match.
    # Frames above this score are prioritized when selecting candidates.
    choice_preferred_clip_score: float = 0.28
