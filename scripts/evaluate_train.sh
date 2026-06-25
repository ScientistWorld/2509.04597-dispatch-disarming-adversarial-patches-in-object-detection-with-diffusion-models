#!/bin/bash
# Visible train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: no method argument is required, and scoring is
# based on saved image artifacts under outputs/ plus data/synthetic/.

set -e

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py --outputs-dir outputs --data-dir data/synthetic --score-path scoring/scores_train.json
