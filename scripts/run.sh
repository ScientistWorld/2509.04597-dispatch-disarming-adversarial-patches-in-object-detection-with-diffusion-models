#!/bin/bash
# Lightweight scorer validation job submitted via action.yaml.
#
# Uses the existing artifacts from the completed reproduction run, refreshes
# the train/test score files with the corrected deterministic split, and checks
# the complete scoring contract.

set -e

cd /home/user

bash scripts/evaluate.sh
bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
python validate.py --compare
