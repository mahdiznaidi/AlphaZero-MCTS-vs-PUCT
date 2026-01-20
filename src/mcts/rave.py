from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAVEConfig:
    enabled: bool = False
    beta: float = 0.5


class RAVE:
    def __init__(self, config: RAVEConfig | None = None) -> None:
        self.config = config or RAVEConfig()

    def info(self) -> str:
        return "RAVE placeholder (not implemented)."
