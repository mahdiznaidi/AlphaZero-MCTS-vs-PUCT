from __future__ import annotations

import argparse
import pathlib
import random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import yaml

from src.game.breakthrough import BreakthroughState
from src.mcts.puct import PUCT
from src.mcts.uct import FlatMonteCarlo, UCT
from src.nn.model import PolicyValueNet
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
    policy_value_fn = None
    if model_path and model_path.exists():
        model = PolicyValueNet(board_size=state.board.shape[0])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        policy_value_fn = model.predict
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


def play_match(agent_a: AgentFn, agent_b: AgentFn, max_moves: int, board_size: int) -> int:
    state = BreakthroughState.initial(board_size=board_size)
    for _ in range(max_moves):
        agent = agent_a if state.player == 1 else agent_b
        move = agent(state)
        state = state.apply_move(move)
        winner = state.winner()
        if winner is not None:
            return winner
    return 0


def run_arena(config: dict, matches: int | None = None) -> ArenaResult:
    set_seed(config["seed"])
    matches = matches or config["arena"]["matches"]
    agent_a = build_agent(config["arena"]["agent_a"], config)
    agent_b = build_agent(config["arena"]["agent_b"], config)
    wins_a = 0
    wins_b = 0
    draws = 0
    for _ in range(matches):
        winner = play_match(agent_a, agent_b, config["max_moves"], config["board_size"])
        if winner == 1:
            wins_a += 1
        elif winner == -1:
            wins_b += 1
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
