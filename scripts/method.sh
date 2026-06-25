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
EPOCHS="${EPOCHS:-8}"
STEPS="${STEPS:-20}"
GRIDS="${GRIDS:-8}"

python data/synthetic_dispatch_data.py --output-dir data/synthetic --train-count "${TRAIN_COUNT:-1200}" --test-count "${TEST_COUNT:-120}"
python method/train_diffusion.py --data-dir data/synthetic/train/clean --checkpoint checkpoints/dispatch_ddpm.pt --epochs "$EPOCHS" --device "$DEVICE"
bash scripts/baseline.sh
python method/dispatch.py --checkpoint checkpoints/dispatch_ddpm.pt --data-dir data/synthetic --outputs-dir outputs --variant hiding --steps "$STEPS" --grids "$GRIDS" --device "$DEVICE"
python method/dispatch.py --checkpoint checkpoints/dispatch_ddpm.pt --data-dir data/synthetic --outputs-dir outputs --variant creating --steps "$STEPS" --grids "$GRIDS" --device "$DEVICE"
bash scripts/evaluate.sh
