#!/bin/bash
# CPU validation job submitted via action.yaml.
#
# This re-evaluates existing DISPATCH image artifacts and validates the
# scoring contract after method/eval separation cleanup.

set -e

cd /home/user

bash scripts/evaluate.sh
python validate.py --compare
