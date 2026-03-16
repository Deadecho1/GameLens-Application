from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class WindowResult:
    start_time: float
    end_time: float
    center_time: float
    scores: Dict[str, float]


@dataclass(frozen=True)
class PeakEvent:
    label: str
    time: float
    score: float
    start_time: float
    end_time: float
    support_windows: Tuple[WindowResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RefinedEvent:
    """A PeakEvent that has been refined to an exact frame/time position."""

    source: PeakEvent
    refined_time: float
    refined_frame: int
    refined_score: float
    refined_window_start: float
    refined_window_end: float
    retry_frames: Tuple[int, ...] = ()
    retry_times: Tuple[float, ...] = ()
    refinement_method: str = ""

    # Convenience accessors delegating to source
    @property
    def label(self) -> str:
        return self.source.label

    @property
    def time(self) -> float:
        return self.source.time

    @property
    def score(self) -> float:
        return self.source.score


@dataclass
class RunCandidate:
    start_event: PeakEvent
    end_event: PeakEvent
    start_time: float
    end_time: float
    score: float


@dataclass
class DecodedRun:
    """Intermediate run with coarse PeakEvent boundaries (pre-refinement)."""

    start: PeakEvent
    end: PeakEvent
    choices: List[PeakEvent]
    drops: List[PeakEvent]


@dataclass(frozen=True)
class RefinedRun:
    """Final run with all events refined to exact frame positions."""

    start: RefinedEvent
    end: RefinedEvent
    choices: Tuple[RefinedEvent, ...] = ()
    drops: Tuple[RefinedEvent, ...] = ()
