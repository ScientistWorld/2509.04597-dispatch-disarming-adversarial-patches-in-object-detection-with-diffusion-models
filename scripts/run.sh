#!/bin/bash
# CPU validation job submitted via action.yaml.
#
# This re-evaluates existing DISPATCH image artifacts, including the
# regenerated-only rectification ablation, and validates the scoring contract.

set -e

cd /home/user

bash scripts/evaluate.sh
python validate.py --compare
