#!/bin/bash
# Split scoring job submitted via action.yaml.
#
# Scores the visible train slice and held-out test slice against the saved
# final artifacts. This intentionally does not retrain or regenerate methods.

set -e

cd /home/user

bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
