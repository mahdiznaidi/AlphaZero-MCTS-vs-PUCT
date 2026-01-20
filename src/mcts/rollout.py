from __future__ import annotations

import random

from src.game.breakthrough import BreakthroughState


def random_rollout(state: BreakthroughState, max_depth: int = 50) -> float:
    current = state.clone()
    for _ in range(max_depth):
        winner = current.winner()
        if winner is not None:
            return 1.0 if winner == state.player else -1.0
        moves = current.legal_moves()
        if not moves:
            return -1.0
        move = random.choice(moves)
        current = current.apply_move(move)
    winner = current.winner()
    if winner is None:
        return 0.0
    return 1.0 if winner == state.player else -1.0
