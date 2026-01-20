from __future__ import annotations

"""
Evaluation arena for playing matches between agents.

Agents are callables taking a :class:`BreakthroughState` and returning a
move.  The arena plays a fixed number of matches and reports the
results.  To ensure fairness with respect to the first‑move advantage,
the arena alternates which agent plays as white (player 1) and which
plays as black (player ‑1) on each game.
"""

import argparse
import pathlib
import random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import yaml

from src.game.breakthrough import BreakthroughState
from src.mcts.puct import PUCT
from src.mcts.uct import FlatMonteCarlo, UCT
# We avoid importing PolicyValueNet at the module level because it pulls in
# the torch dependency.  Torch is only needed when evaluating with a
# trained neural network, so we lazily import both torch and
# PolicyValueNet inside ``puct_agent`` when a model path is specified.
from src.utils.metrics import win_rate
from src.utils.seed import set_seed


AgentFn = Callable[[BreakthroughState], Tuple[int, int, int, int]]


@dataclass
class ArenaResult:
    wins_a: int
    wins_b: int
    draws: int


def random_agent(state: BreakthroughState) -> Tuple[int, int, int, int]:
    moves = state.legal_moves()
    if not moves:
        raise ValueError("No legal moves")
    return random.choice(moves)


def flat_mc_agent(state: BreakthroughState, rollouts: int, rollout_depth: int) -> Tuple[int, int, int, int]:
    agent = FlatMonteCarlo(rollouts_per_move=rollouts, rollout_depth=rollout_depth)
    return agent.select_move(state)


def uct_agent(state: BreakthroughState, simulations: int, c_uct: float, rollout_depth: int) -> Tuple[int, int, int, int]:
    agent = UCT(simulations=simulations, c_uct=c_uct, rollout_depth=rollout_depth)
    return agent.search(state)


def puct_agent(state: BreakthroughState, simulations: int, c_puct: float, rollout_depth: int, model_path: pathlib.Path | None) -> Tuple[int, int, int, int]:
    """
    Select a move using the PUCT algorithm.  If a trained model is provided,
    it will be used to guide the search and evaluate leaf nodes; otherwise
    the search falls back to a uniform prior and rollout evaluations.

    Args:
        state: Current game state.
        simulations: Number of simulations to run.
        c_puct: Exploration constant for PUCT.
        rollout_depth: Maximum depth for random rollouts.
        model_path: Optional path to a trained neural network.

    Returns:
        A move (row_from, col_from, row_to, col_to).
    """
    policy_value_fn = None
    # Lazily import torch and PolicyValueNet only if a model is specified and exists.
    if model_path and model_path.exists():
        try:
            import torch  # type: ignore
            from src.nn.model import PolicyValueNet
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is required for using a trained neural network with PUCT."
            ) from exc
        # Instantiate the model and load weights
        model = PolicyValueNet(board_size=state.board.shape[0])
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        policy_value_fn = model.predict
    # Create the PUCT searcher with or without the policy/value function
    agent = PUCT(
        simulations=simulations,
        c_puct=c_puct,
        rollout_depth=rollout_depth,
        policy_value_fn=policy_value_fn,
        board_size=state.board.shape[0],
    )
    return agent.search(state)


def build_agent(name: str, config: dict) -> AgentFn:
    name = name.lower()
    if name == "random":
        return random_agent
    if name == "flat":
        return lambda state: flat_mc_agent(state, rollouts=10, rollout_depth=config["mcts"]["rollout_depth"])
    if name == "uct":
        return lambda state: uct_agent(
            state,
            simulations=config["mcts"]["simulations"],
            c_uct=config["mcts"].get("c_uct", 1.4),
            rollout_depth=config["mcts"]["rollout_depth"],
        )
    if name == "puct":
        model_path = pathlib.Path(config.get("model", {}).get("path", ""))
        return lambda state: puct_agent(
            state,
            simulations=config["mcts"]["simulations"],
            c_puct=config["mcts"]["c_puct"],
            rollout_depth=config["mcts"]["rollout_depth"],
            model_path=model_path,
        )
    raise ValueError(f"Unknown agent: {name}")


def play_match(agent_white: AgentFn, agent_black: AgentFn, max_moves: int, board_size: int) -> int:
    """Play a single match between two agents.

    Args:
        agent_white: Agent playing as white (player 1).
        agent_black: Agent playing as black (player ‑1).
        max_moves: Move limit to avoid infinite games.
        board_size: Size of the board.

    Returns:
        1 if white wins, ‑1 if black wins, 0 for a draw.
    """
    state = BreakthroughState.initial(board_size=board_size)
    for _ in range(max_moves):
        # Pick the agent based on the current player
        if state.player == 1:
            move = agent_white(state)
        else:
            move = agent_black(state)
        state = state.apply_move(move)
        winner = state.winner()
        if winner is not None:
            return winner
    return 0


def run_arena(config: dict, matches: int | None = None) -> ArenaResult:
    """Run an arena of matches between two agents.

    Agents alternate playing white and black to mitigate first player
    advantage.  Results are aggregated as wins, losses and draws for
    agent A (the agent specified as ``agent_a`` in the config).

    Args:
        config: Dictionary loaded from a YAML config file.
        matches: Optional override for the number of matches.

    Returns:
        An :class:`ArenaResult` summarising the results.
    """
    set_seed(config["seed"])
    matches = matches or config["arena"]["matches"]
    agent_a = build_agent(config["arena"]["agent_a"], config)
    agent_b = build_agent(config["arena"]["agent_b"], config)
    wins_a = 0
    wins_b = 0
    draws = 0
    for i in range(matches):
        # On even games, agent_a plays white; on odd games, agent_b plays white
        if i % 2 == 0:
            winner = play_match(agent_a, agent_b, config["max_moves"], config["board_size"])
            # winner refers to the colour: 1=white, -1=black
            if winner == 1:
                wins_a += 1
            elif winner == -1:
                wins_b += 1
            else:
                draws += 1
        else:
            winner = play_match(agent_b, agent_a, config["max_moves"], config["board_size"])
            # Map result back to agent_a vs agent_b perspective
            if winner == 1:
                wins_b += 1  # agent_b was white
            elif winner == -1:
                wins_a += 1  # agent_a was black
            else:
                draws += 1
    print(f"Agent A ({config['arena']['agent_a']}) wins: {wins_a}")
    print(f"Agent B ({config['arena']['agent_b']}) wins: {wins_b}")
    print(f"Draws: {draws}")
    print(f"Win rate A: {win_rate(wins_a, wins_b):.2f}")
    return ArenaResult(wins_a=wins_a, wins_b=wins_b, draws=draws)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation matches")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--matches", type=int, help="Override matches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_arena(config, matches=args.matches)


if __name__ == "__main__":
    main()
