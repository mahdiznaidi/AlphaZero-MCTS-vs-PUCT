#!/usr/bin/env bash
set -euo pipefail

python -m src.selfplay.arena --config configs/uct.yaml
