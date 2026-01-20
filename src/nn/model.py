from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.action_space import ActionSpace
from src.game.encoding import encode_state
from src.game.breakthrough import BreakthroughState


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = 5, channels: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_space = ActionSpace(board_size=board_size)
        self.backbone = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.action_space.num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value.squeeze(-1)

    @torch.no_grad()
    def predict(self, state: BreakthroughState) -> Tuple[np.ndarray, float]:
        self.eval()
        encoded = encode_state(state.board, state.player)
        tensor = torch.from_numpy(encoded).unsqueeze(0)
        policy_logits, value = self.forward(tensor)
        policy = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        mask = self.action_space.legal_action_mask(state.board, state.player)
        masked = policy * mask
        if masked.sum() <= 0:
            masked = mask
        if masked.sum() > 0:
            masked = masked / masked.sum()
        return masked.astype(np.float32), float(value.item())


@dataclass
class TrainingBatch:
    states: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
