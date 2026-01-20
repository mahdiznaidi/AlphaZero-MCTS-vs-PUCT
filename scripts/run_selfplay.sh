#!/usr/bin/env bash
set -euo pipefail

python -m src.selfplay.generate --config configs/train.yaml
