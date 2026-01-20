from __future__ import annotations

import math
import random
from typing import Dict, Tuple

from src.game.breakthrough import BreakthroughState
from src.mcts.node import Node
from src.mcts.rollout import random_rollout


class UCT:
    def __init__(self, simulations: int = 100, c_uct: float = 1.4, rollout_depth: int = 50) -> None:
        self.simulations = simulations
        self.c_uct = c_uct
        self.rollout_depth = rollout_depth

    def search(self, state: BreakthroughState) -> Tuple[int, int, int, int]:
        root = Node(state=state)
        for _ in range(self.simulations):
            node = root
            while node.expanded() and not node.state.is_terminal():
                node = self._select_child(node)
            if not node.state.is_terminal():
                self._expand(node)
                if node.expanded():
                    node = random.choice(list(node.children.values()))
            value = random_rollout(node.state, max_depth=self.rollout_depth)
            self._backpropagate(node, value)
        if not root.children:
            moves = state.legal_moves()
            return random.choice(moves)
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def _select_child(self, node: Node) -> Node:
        total = math.log(node.visit_count + 1)
        best_score = -float("inf")
        best_child = None
        for move, child in node.children.items():
            exploit = child.value()
            explore = self.c_uct * math.sqrt(total / (child.visit_count + 1))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child if best_child is not None else random.choice(list(node.children.values()))

    def _expand(self, node: Node) -> None:
        moves = node.state.legal_moves()
        for move in moves:
            if move not in node.children:
                node.children[move] = Node(state=node.state.apply_move(move), parent=node)

    def _backpropagate(self, node: Node, value: float) -> None:
        current = node
        current_value = value
        while current is not None:
            current.visit_count += 1
            current.value_sum += current_value
            current_value = -current_value
            current = current.parent


class FlatMonteCarlo:
    def __init__(self, rollouts_per_move: int = 10, rollout_depth: int = 50) -> None:
        self.rollouts_per_move = rollouts_per_move
        self.rollout_depth = rollout_depth

    def select_move(self, state: BreakthroughState) -> Tuple[int, int, int, int]:
        moves = state.legal_moves()
        if not moves:
            raise ValueError("No legal moves")
        scores: Dict[Tuple[int, int, int, int], float] = {}
        for move in moves:
            total = 0.0
            for _ in range(self.rollouts_per_move):
                value = random_rollout(state.apply_move(move), max_depth=self.rollout_depth)
                total += -value
            scores[move] = total / self.rollouts_per_move
        return max(scores.items(), key=lambda item: item[1])[0]
