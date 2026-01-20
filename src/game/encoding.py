from __future__ import annotations

import numpy as np


def encode_state(board: np.ndarray, player: int, include_player_plane: bool = True) -> np.ndarray:
    current = (board == player).astype(np.float32)
    opponent = (board == -player).astype(np.float32)
    planes = [current, opponent]
    if include_player_plane:
        planes.append(np.full_like(current, fill_value=player, dtype=np.float32))
    return np.stack(planes, axis=0)
