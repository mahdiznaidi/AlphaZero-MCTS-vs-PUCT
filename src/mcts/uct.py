from __future__ import annotations

"""
Implementation of UCT (Upper Confidence Bounds for Trees) and a flat
Monte‑Carlo baseline for Breakthrough.

The UCT algorithm builds a search tree guided by the UCT formula
``Q(s,a) + c * sqrt(log(N(s)) / (N(s,a) + 1))``.  It uses the random
rollout defined in :mod:`src.mcts.rollout` for leaf evaluation.  Values
are accumulated relative to the root player; sign flips are applied
during backpropagation to maintain the zero‑sum property.
"""

import math
import random
from typing import Dict, Tuple

from src.game.breakthrough import BreakthroughState
from src.mcts.node import Node
from src.mcts.rollout import random_rollout


class UCT:
    """A simple Monte‑Carlo Tree Search using the UCT formula."""

    def __init__(self, simulations: int = 100, c_uct: float = 1.4, rollout_depth: int = 50) -> None:
        self.simulations = simulations
        self.c_uct = c_uct
        self.rollout_depth = rollout_depth

    def search(self, state: BreakthroughState) -> Tuple[int, int, int, int]:
        """Run a UCT search from the given state and return the best move."""
        root = Node(state=state)
        for _ in range(self.simulations):
            node = root
            # Selection: descend the tree while fully expanded
            while node.expanded() and not node.state.is_terminal():
                node = self._select_child(node)
            # Expansion: add children if this is a non‑terminal leaf
            if not node.state.is_terminal():
                self._expand(node)
                if node.expanded():
                    node = random.choice(list(node.children.values()))
            # Simulation: evaluate the leaf via random rollout
            value = random_rollout(node.state, max_depth=self.rollout_depth)
            # Backpropagation: propagate value up the tree
            self._backpropagate(node, value)
        # If the root has no children (no legal moves), fall back to random
        if not root.children:
            moves = state.legal_moves()
            return random.choice(moves)
        # Choose the move with the highest visit count
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def _select_child(self, node: Node) -> Node:
        """Select a child node using the UCT score.

        The stored value at each child is from the perspective of the player to
        move at that child.  However, when selecting a move for the current
        player (represented by ``node.state.player``), we need to evaluate
        each child from the current player's perspective.  Since the child
        represents the position after the current player has made a move,
        the value returned by :meth:`Node.value` must be negated to convert
        from the child player's viewpoint to the current player's viewpoint.
        """
        total = math.log(node.visit_count + 1)
        best_score = -float("inf")
        best_child = None
        for move, child in node.children.items():
            # Exploitation term: negate the child's value because it is
            # stored from the child player's perspective.  A positive value
            # means good for the child (opponent), so it is bad for the
            # current player.
            exploit = -child.value()
            # Exploration term: same as standard UCT
            explore = self.c_uct * math.sqrt(total / (child.visit_count + 1))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        # Should not happen, but return a random child as a fallback
        return best_child if best_child is not None else random.choice(list(node.children.values()))

    def _expand(self, node: Node) -> None:
        """Expand a node by generating all legal children."""
        moves = node.state.legal_moves()
        for move in moves:
            if move not in node.children:
                node.children[move] = Node(state=node.state.apply_move(move), parent=node)

    def _backpropagate(self, node: Node, value: float) -> None:
        """Propagate the rollout value up the tree, flipping sign at each level."""
        current = node
        current_value = value
        while current is not None:
            current.visit_count += 1
            current.value_sum += current_value
            current_value = -current_value
            current = current.parent


class FlatMonteCarlo:
    """Flat Monte‑Carlo baseline for selecting moves."""

    def __init__(self, rollouts_per_move: int = 10, rollout_depth: int = 50) -> None:
        self.rollouts_per_move = rollouts_per_move
        self.rollout_depth = rollout_depth

    def select_move(self, state: BreakthroughState) -> Tuple[int, int, int, int]:
        """Evaluate each legal move by independent random rollouts and return the best.

        Args:
            state: The current position.

        Returns:
            The move with the highest average rollout value from the root player's perspective.
        """
        moves = state.legal_moves()
        if not moves:
            raise ValueError("No legal moves")
        scores: Dict[Tuple[int, int, int, int], float] = {}
        for move in moves:
            total = 0.0
            for _ in range(self.rollouts_per_move):
                value = random_rollout(state.apply_move(move), max_depth=self.rollout_depth)
                # Flip the value because after we apply the move, the perspective switches
                total += -value
            scores[move] = total / self.rollouts_per_move
        # Pick the move with the highest average score
        return max(scores.items(), key=lambda item: item[1])[0]
