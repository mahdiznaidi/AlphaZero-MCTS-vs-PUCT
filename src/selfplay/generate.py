from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import yaml

from src.game.action_space import ActionSpace
from src.game.breakthrough import BreakthroughState
from src.game.encoding import encode_state
from src.mcts.puct import PUCT
from src.nn.model import PolicyValueNet
from src.utils.seed import set_seed


@dataclass
class SelfPlaySample:
    state: np.ndarray
    policy: np.ndarray
    value: float
    player: int


def temperature_policy(policy: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return policy
    scaled = np.power(policy, 1.0 / temperature)
    if scaled.sum() == 0:
        return policy
    return scaled / scaled.sum()


def play_game(mcts: PUCT, config: dict) -> List[SelfPlaySample]:
    state = BreakthroughState.initial(board_size=config["board_size"])
    samples: List[SelfPlaySample] = []
    temperature_moves = config["selfplay"]["temperature_moves"]
    for move_index in range(config["max_moves"]):
        policy = mcts.root_policy(state)
        temperature = config["selfplay"]["temperature_start"] if move_index < temperature_moves else config["selfplay"]["temperature_end"]
        policy = temperature_policy(policy, temperature)
        action_space = ActionSpace(board_size=config["board_size"])
        moves = action_space.moves_from_policy(state.board, state.player, policy)
        if not moves:
            break
        moves_list = list(moves.items())
        move_probs = np.array([prob for _, prob in moves_list], dtype=np.float32)
        move_probs = move_probs / move_probs.sum()
        chosen_index = np.random.choice(len(moves_list), p=move_probs)
        move = moves_list[chosen_index][0]
        samples.append(
            SelfPlaySample(
                state=encode_state(state.board, state.player),
                policy=policy,
                value=0.0,
                player=state.player,
            )
        )
        state = state.apply_move(move)
        if state.is_terminal():
            break
    winner = state.winner()
    for sample in samples:
        if winner is None:
            sample.value = 0.0
        else:
            sample.value = 1.0 if winner == sample.player else -1.0
    return samples


def generate_selfplay(config: dict) -> None:
    set_seed(config["seed"])
    model_path = pathlib.Path(config.get("model", {}).get("path", ""))
    policy_value_fn = None
    if model_path and model_path.exists():
        model = PolicyValueNet(board_size=config["board_size"])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        policy_value_fn = model.predict

    mcts = PUCT(
        simulations=config["mcts"]["simulations"],
        c_puct=config["mcts"]["c_puct"],
        rollout_depth=config["mcts"]["rollout_depth"],
        policy_value_fn=policy_value_fn,
        board_size=config["board_size"],
    )

    all_samples: List[SelfPlaySample] = []
    for _ in range(config["selfplay"]["games"]):
        all_samples.extend(play_game(mcts, config))

    states = np.stack([s.state for s in all_samples], axis=0)
    policies = np.stack([s.policy for s in all_samples], axis=0)
    values = np.array([s.value for s in all_samples], dtype=np.float32)

    output_path = pathlib.Path(config["selfplay"]["dataset_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, states=states, policies=policies, values=values)
    print(f"Saved dataset to {output_path} with {len(all_samples)} samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate self-play games")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--games", type=int, help="Override number of games")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.games is not None:
        config["selfplay"]["games"] = args.games
    generate_selfplay(config)


if __name__ == "__main__":
    main()
