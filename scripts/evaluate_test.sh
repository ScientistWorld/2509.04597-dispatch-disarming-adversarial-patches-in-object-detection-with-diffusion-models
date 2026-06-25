#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: no method argument is required. When
# scoring/scores_train.json exists, the test slice scores every method entry
# present there so train/test comparisons use matching method keys.

set -e

cd /home/user

mkdir -p scoring

python eval/test/evaluate.py \
  --outputs-dir outputs \
  --data-dir data/synthetic \
  --score-path scoring/scores_test.json \
  --train-score-path scoring/scores_train.json
