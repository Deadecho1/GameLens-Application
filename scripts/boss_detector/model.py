from dataclasses import dataclass


@dataclass(frozen=True)
class BossDetectionResult:
    class_name: str
    confidence: float
