from __future__ import annotations

"""
Implementation of PUCT (Predictor + UCT) for Breakthrough.

This class implements a variant of Monte‑Carlo tree search similar to
AlphaZero's search algorithm.  It maintains prior probabilities and a
value estimate supplied by a policy/value network.  When no network
function is provided, the algorithm falls back to a uniform prior and
pure tree search.

The code is adapted from the original project but kept mostly intact
for comparability.  See the UCT implementation in :mod:`src.mcts.uct`
for comments on common MCTS components.
"""

import math
import random
from typing import Callable, Dict, Tuple

import numpy as np

from src.game.action_space import ActionSpace
from src.game.breakthrough import BreakthroughState
from src.mcts.node import Node

PolicyValueFn = Callable[[BreakthroughState], Tuple[np.ndarray, float]]


class PUCT:
    def __init__(
        self,
        simulations: int = 100,
        c_puct: float = 1.5,
        rollout_depth: int = 50,
        policy_value_fn: PolicyValueFn | None = None,
        board_size: int = 5,
    ) -> None:
        self.simulations = simulations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.policy_value_fn = policy_value_fn
        self.action_space = ActionSpace(board_size=board_size)

    def search(self, state: BreakthroughState) -> Tuple[int, int, int, int]:
        root = Node(state=state)
        # Always expand the root once so that priors are assigned
        self._expand(root)
        for _ in range(self.simulations):
            node = root
            search_path = [node]
            # Selection
            while node.expanded() and not node.state.is_terminal():
                node = self._select_child(node)
                search_path.append(node)
            # Evaluation (terminal or network)
            value = self._evaluate(node)
            # Backpropagation
            self._backpropagate(search_path, value)
        # Fallback to random if no moves
        if not root.children:
            moves = state.legal_moves()
            return random.choice(moves)
        # Select move with highest visit count
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def root_policy(self, state: BreakthroughState) -> np.ndarray:
        """Return a policy distribution over actions at the root.

        This method runs a search identical to :meth:`search` but
        returns the visit counts as a probability distribution over the
        fixed action space.  Illegal actions get probability zero.  If
        there are no legal actions, a zero vector is returned.
        """
        root = Node(state=state)
        self._expand(root)
        for _ in range(self.simulations):
            node = root
            search_path = [node]
            while node.expanded() and not node.state.is_terminal():
                node = self._select_child(node)
                search_path.append(node)
            value = self._evaluate(node)
            self._backpropagate(search_path, value)
        policy = np.zeros(self.action_space.num_actions, dtype=np.float32)
        for move, child in root.children.items():
            action = self._move_to_action(move, state.player)
            if action is None:
                continue
            policy[self.action_space.action_to_index(action)] = child.visit_count
        if policy.sum() > 0:
            policy = policy / policy.sum()
        return policy

    def _select_child(self, node: Node) -> Node:
        """
        Select a child node using the PUCT formula.

        The stored ``value`` on each child is from the viewpoint of the
        player to move at that child.  When selecting from ``node``, we need
        to evaluate each child from the perspective of ``node.state.player``,
        i.e., the current player.  Therefore the exploitation term is the
        negation of ``child.value()``.  The exploration term is computed
        according to the prior probability and the PUCT constant.
        """
        best_score = -float("inf")
        best_child = None
        total_visits = math.sqrt(node.visit_count + 1)
        for child in node.children.values():
            prior = child.prior
            # Negate value because child.value() is from the child's
            # perspective (opponent of the current player)
            exploit = -child.value()
            explore = self.c_puct * prior * total_visits / (1 + child.visit_count)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child if best_child is not None else random.choice(list(node.children.values()))

    def _evaluate(self, node: Node) -> float:
        winner = node.state.winner()
        if winner is not None:
            # Leaf is terminal
            return 1.0 if winner == node.state.player else -1.0
        if self.policy_value_fn is None:
            # Without a network, use a value of zero (rollouts could be plugged in here)
            return 0.0
        _, value = self.policy_value_fn(node.state)
        return float(value)

    def _expand(self, node: Node) -> None:
        if node.state.is_terminal():
            return
        # Get prior probabilities from network or uniform
        if self.policy_value_fn is None:
            policy = np.ones(self.action_space.num_actions, dtype=np.float32)
        else:
            policy, _ = self.policy_value_fn(node.state)
        moves_with_probs = self.action_space.moves_from_policy(node.state.board, node.state.player, policy)
        for move, prob in moves_with_probs.items():
            if move not in node.children:
                node.children[move] = Node(state=node.state.apply_move(move), parent=node, prior=prob)

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        current_value = value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value

    def _move_to_action(self, move: Tuple[int, int, int, int], player: int) -> Tuple[int, int, int] | None:
        # Convert a move (r0,c0,r1,c1) to an action (row,col,direction) relative to the player
        r0, c0, r1, c1 = move
        delta_row = -1 if player == 1 else 1
        if r1 - r0 != delta_row:
            return None
        if c1 == c0:
            return (r0, c0, 0)
        if c1 == c0 - 1:
            return (r0, c0, 1)
        if c1 == c0 + 1:
            return (r0, c0, 2)
        return None
