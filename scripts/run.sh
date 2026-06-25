#!/bin/bash
# Evaluation validation job submitted via action.yaml.
#
# Scores the existing DISPATCH artifacts, including the main score file and
# train/test split score files, then validates the scoring contract.

set -e

cd /home/user

bash scripts/evaluate.sh
bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
python validate.py --compare
