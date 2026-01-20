from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.game.action_space import ActionSpace
from src.nn.model import PolicyValueNet
from src.utils.seed import set_seed


def load_dataset(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["states"], data["policies"], data["values"]


def train(config: dict, dataset_path: pathlib.Path | None = None) -> None:
    set_seed(config["seed"])
    training_cfg = config["training"]
    model = PolicyValueNet(board_size=config["board_size"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    dataset = dataset_path or pathlib.Path(training_cfg.get("dataset_path", config["selfplay"]["dataset_path"]))
    states, policies, values = load_dataset(dataset)
    states = torch.from_numpy(states)
    policies = torch.from_numpy(policies)
    values = torch.from_numpy(values)

    batch_size = training_cfg["batch_size"]
    epochs = training_cfg["epochs"]
    total = states.shape[0]

    for epoch in range(epochs):
        indices = torch.randperm(total)
        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            batch_states = states[idx]
            batch_policies = policies[idx]
            batch_values = values[idx]
            logits, value_pred = model(batch_states)
            policy_loss = -(batch_policies * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            value_loss = F.mse_loss(value_pred, batch_values)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} - loss: {loss.item():.4f}")

    model_path = pathlib.Path(training_cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy/value network")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dataset", help="Override dataset path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset = pathlib.Path(args.dataset) if args.dataset else None
    train(config, dataset)


if __name__ == "__main__":
    main()
