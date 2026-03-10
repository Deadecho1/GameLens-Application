from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class EventPoint:
    time: Optional[float]
    frame: Optional[int]
    confidence: Optional[float]


@dataclass
class RunEventJson:
    run_index: int
    start_event: EventPoint
    end_event: EventPoint
    choice_events: List[EventPoint] = field(default_factory=list)
    drop_events: List[EventPoint] = field(default_factory=list)


@dataclass
class VideoEventsJson:
    video_name: str
    fps: float
    duration_seconds: float
    runs: List[RunEventJson]


@dataclass
class RunChoiceResult:
    options: Any
    selected_choice: Any


@dataclass
class ExportedRun:
    start_time: Optional[float]
    end_time: Optional[float]
    duration: Optional[float]
    choices: List[dict]