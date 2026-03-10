from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WindowResult:
    start_time: float
    end_time: float
    center_time: float
    scores: Dict[str, float]


@dataclass
class PeakEvent:
    label: str
    time: float
    score: float
    start_time: float
    end_time: float
    support_windows: List[WindowResult] = field(default_factory=list)

    refined_time: Optional[float] = None
    refined_frame: Optional[int] = None
    refined_score: Optional[float] = None
    refined_window_start: Optional[float] = None
    refined_window_end: Optional[float] = None


@dataclass
class RunCandidate:
    start_event: PeakEvent
    end_event: PeakEvent
    start_time: float
    end_time: float
    score: float


@dataclass
class DecodedRun:
    start: PeakEvent
    end: PeakEvent
    choices: List[PeakEvent]
    drops: List[PeakEvent]