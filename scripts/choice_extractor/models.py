from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ExtractionResult:
    choices: List[str]
    selected_choice: Optional[str]
