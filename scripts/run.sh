#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

export DEVICE="${DEVICE:-cpu}"
export TRAIN_COUNT="${TRAIN_COUNT:-1200}"
export TEST_COUNT="${TEST_COUNT:-80}"
export EPOCHS="${EPOCHS:-20}"
export STEPS="${STEPS:-24}"
export GRIDS="${GRIDS:-8}"

bash scripts/reproduce.sh
