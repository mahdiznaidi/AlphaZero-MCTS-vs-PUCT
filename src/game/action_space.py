from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Action = Tuple[int, int, int]


@dataclass(frozen=True)
class ActionSpace:
    board_size: int = 5

    def __post_init__(self) -> None:
        if self.board_size <= 0:
            raise ValueError("board_size must be positive")

    @property
    def num_actions(self) -> int:
        return self.board_size * self.board_size * 3

    def index_to_action(self, index: int) -> Action:
        if index < 0 or index >= self.num_actions:
            raise ValueError("action index out of range")
        cell = index // 3
        direction = index % 3
        row = cell // self.board_size
        col = cell % self.board_size
        return row, col, direction

    def action_to_index(self, action: Action) -> int:
        row, col, direction = action
        return (row * self.board_size + col) * 3 + direction

    def action_to_move(self, action: Action, player: int) -> Tuple[int, int, int, int]:
        row, col, direction = action
        delta_row = 1 if player == 1 else -1
        if direction == 0:
            return row, col, row + delta_row, col
        if direction == 1:
            return row, col, row + delta_row, col - 1
        if direction == 2:
            return row, col, row + delta_row, col + 1
        raise ValueError("invalid direction")

    def legal_action_mask(self, board: np.ndarray, player: int) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.float32)
        rows, cols = board.shape
        opponent = -player
        for row in range(rows):
            for col in range(cols):
                if board[row, col] != player:
                    continue
                forward_row = row + (1 if player == 1 else -1)
                if forward_row < 0 or forward_row >= rows:
                    continue
                if board[forward_row, col] == 0:
                    mask[self.action_to_index((row, col, 0))] = 1.0
                if col - 1 >= 0 and board[forward_row, col - 1] == opponent:
                    mask[self.action_to_index((row, col, 1))] = 1.0
                if col + 1 < cols and board[forward_row, col + 1] == opponent:
                    mask[self.action_to_index((row, col, 2))] = 1.0
        return mask

    def moves_from_policy(self, board: np.ndarray, player: int, policy: np.ndarray) -> Dict[Tuple[int, int, int, int], float]:
        mask = self.legal_action_mask(board, player)
        masked = policy * mask
        if masked.sum() == 0:
            masked = mask
        if masked.sum() == 0:
            return {}
        normalized = masked / masked.sum()
        moves: Dict[Tuple[int, int, int, int], float] = {}
        for idx, prob in enumerate(normalized):
            if prob <= 0:
                continue
            action = self.index_to_action(idx)
            move = self.action_to_move(action, player)
            moves[move] = float(prob)
        return moves

    def moves_from_uniform(self, board: np.ndarray, player: int) -> Dict[Tuple[int, int, int, int], float]:
        mask = self.legal_action_mask(board, player)
        if mask.sum() == 0:
            return {}
        probs = mask / mask.sum()
        moves: Dict[Tuple[int, int, int, int], float] = {}
        for idx, prob in enumerate(probs):
            if prob <= 0:
                continue
            action = self.index_to_action(idx)
            move = self.action_to_move(action, player)
            moves[move] = float(prob)
        return moves
