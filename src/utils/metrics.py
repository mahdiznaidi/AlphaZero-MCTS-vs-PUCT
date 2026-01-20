from __future__ import annotations


def win_rate(wins_a: int, wins_b: int) -> float:
    total = wins_a + wins_b
    if total == 0:
        return 0.0
    return wins_a / total
