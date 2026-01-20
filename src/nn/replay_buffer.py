from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class ReplaySample:
    state: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, max_size: int = 50000) -> None:
        self.max_size = max_size
        self.buffer: Deque[ReplaySample] = deque(maxlen=max_size)

    def add(self, samples: List[ReplaySample]) -> None:
        for sample in samples:
            self.buffer.append(sample)

    def __len__(self) -> int:
        return len(self.buffer)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = np.stack([s.state for s in self.buffer], axis=0)
        policies = np.stack([s.policy for s in self.buffer], axis=0)
        values = np.array([s.value for s in self.buffer], dtype=np.float32)
        return states, policies, values
