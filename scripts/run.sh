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
export TRAIN_COUNT="${TRAIN_COUNT:-2000}"
export TEST_COUNT="${TEST_COUNT:-80}"
export EPOCHS="${EPOCHS:-60}"
export STEPS="${STEPS:-32}"
export GRIDS="${GRIDS:-16}"

bash scripts/reproduce.sh
