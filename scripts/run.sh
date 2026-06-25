#!/bin/bash
# Full CPU rebuild and validation job submitted via action.yaml.
#
# Regenerates the synthetic benchmark, retrains the scaled inpainting model,
# writes fresh DISPATCH artifacts for clean/hiding/creating variants, then
# refreshes all score files and validates the scoring contract.

set -e

cd /home/user

rm -rf outputs checkpoints/dispatch_ddpm.pt
bash scripts/method.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
python validate.py --compare
