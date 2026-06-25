#!/bin/bash
# Example job script submitted via action.yaml.
#
# This runs the full scaled DISPATCH reproduction inside the compute container.

set -e

cd /home/user

export DEVICE="${DEVICE:-cpu}"
export TRAIN_COUNT="${TRAIN_COUNT:-2000}"
export TEST_COUNT="${TEST_COUNT:-80}"
export EPOCHS="${EPOCHS:-60}"
export STEPS="${STEPS:-32}"
export GRIDS="${GRIDS:-16}"

bash scripts/reproduce.sh
