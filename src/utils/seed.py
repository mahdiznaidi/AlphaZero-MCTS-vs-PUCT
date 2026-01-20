from __future__ import annotations

"""
Random seed utilities.

Sets seeds for Python's random, NumPy and PyTorch to ensure reproducible
experiments.
"""

import random

import numpy as np

try:
    # PyTorch may not be installed in some environments (e.g., when only
    # running pure MCTS agents).  Import it optionally so that seeding
    # remains robust without torch.
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Only seed PyTorch if it is available; otherwise silently ignore.
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
