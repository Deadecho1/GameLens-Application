from __future__ import annotations

from dataclasses import dataclass


from typing import List


@dataclass(frozen=True)
class BossNameResult:
    boss_names: List[str]
