#!/bin/bash
# Full CPU reproduction job submitted via action.yaml.
#
# Rebuilds synthetic data, retrains the clean-data inpainting DDPM, runs
# DISPATCH on hiding and creating attacks, evaluates the written artifacts, and
# validates the scoring contract.

set -e

cd /home/user

rm -rf outputs checkpoints data/synthetic

export DEVICE="${DEVICE:-cpu}"
export TRAIN_COUNT="${TRAIN_COUNT:-2000}"
export TEST_COUNT="${TEST_COUNT:-80}"
export EPOCHS="${EPOCHS:-60}"
export GRIDS="${GRIDS:-16}"
export STEPS="${STEPS:-32}"

bash scripts/reproduce.sh
python validate.py --compare
