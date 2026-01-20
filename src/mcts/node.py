from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.game.breakthrough import BreakthroughState


@dataclass
class Node:
    state: BreakthroughState
    parent: Optional["Node"] = None
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[Tuple[int, int, int, int], "Node"] = field(default_factory=dict)

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
