#!/bin/bash
# End-to-end CPU reproduction job submitted via action.yaml.
#
# Rebuilds the scaled DISPATCH benchmark artifacts from source rather than
# relying on ignored outputs/checkpoints already present in this workspace.

set -e

cd /home/user

rm -rf outputs checkpoints

export DEVICE="${DEVICE:-cpu}"
export TRAIN_COUNT="${TRAIN_COUNT:-2000}"
export TEST_COUNT="${TEST_COUNT:-80}"
export EPOCHS="${EPOCHS:-60}"
export GRIDS="${GRIDS:-16}"
export STEPS="${STEPS:-32}"

bash scripts/reproduce.sh
python validate.py --compare
