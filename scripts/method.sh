#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-60}"
STEPS="${STEPS:-32}"
GRIDS="${GRIDS:-16}"
TRAIN_COUNT="${TRAIN_COUNT:-2000}"
TEST_COUNT="${TEST_COUNT:-80}"

python data/synthetic_dispatch_data.py --output-dir data/synthetic --train-count "$TRAIN_COUNT" --test-count "$TEST_COUNT"
python method/train_diffusion.py --data-dir data/synthetic/train/clean --checkpoint checkpoints/dispatch_ddpm.pt --epochs "$EPOCHS" --grids "$GRIDS" --device "$DEVICE"
bash scripts/baseline.sh
python method/dispatch.py --checkpoint checkpoints/dispatch_ddpm.pt --data-dir data/synthetic --outputs-dir outputs --variant hiding --steps "$STEPS" --grids "$GRIDS" --device "$DEVICE"
python method/dispatch.py --checkpoint checkpoints/dispatch_ddpm.pt --data-dir data/synthetic --outputs-dir outputs --variant creating --steps "$STEPS" --grids "$GRIDS" --device "$DEVICE"
bash scripts/evaluate.sh
