from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

Move = Tuple[int, int, int, int]


@dataclass
class BreakthroughState:
    board: np.ndarray
    player: int

    @staticmethod
    def initial(board_size: int = 5) -> "BreakthroughState":
        board = np.zeros((board_size, board_size), dtype=np.int8)
        board[0:2, :] = -1
        board[-2:, :] = 1
        return BreakthroughState(board=board, player=1)

    def clone(self) -> "BreakthroughState":
        return BreakthroughState(board=self.board.copy(), player=self.player)

    def legal_moves(self) -> List[Move]:
        moves: List[Move] = []
        direction = -1 if self.player == -1 else 1
        opponent = -self.player
        rows, cols = self.board.shape
        for r in range(rows):
            for c in range(cols):
                if self.board[r, c] != self.player:
                    continue
                forward_r = r + direction
                if 0 <= forward_r < rows:
                    if self.board[forward_r, c] == 0:
                        moves.append((r, c, forward_r, c))
                    for dc in (-1, 1):
                        diag_c = c + dc
                        if 0 <= diag_c < cols and self.board[forward_r, diag_c] == opponent:
                            moves.append((r, c, forward_r, diag_c))
        return moves

    def apply_move(self, move: Move) -> "BreakthroughState":
        r0, c0, r1, c1 = move
        next_state = self.clone()
        next_state.board[r1, c1] = next_state.board[r0, c0]
        next_state.board[r0, c0] = 0
        next_state.player = -self.player
        return next_state

    def is_terminal(self) -> bool:
        return self.winner() is not None

    def winner(self) -> Optional[int]:
        rows, _ = self.board.shape
        if np.any(self.board[0, :] == 1) or np.any(self.board[-1, :] == -1):
            return 1 if np.any(self.board[0, :] == 1) else -1
        if np.sum(self.board == 1) == 0:
            return -1
        if np.sum(self.board == -1) == 0:
            return 1
        if not self.legal_moves():
            return -self.player
        return None

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for r in range(self.board.shape[0]):
            rows.append(" ".join(symbols[int(v)] for v in self.board[r]))
        return "\n".join(rows)
