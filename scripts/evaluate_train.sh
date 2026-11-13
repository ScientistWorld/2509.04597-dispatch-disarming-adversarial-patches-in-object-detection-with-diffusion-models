#!/bin/bash
# Visible slice.  No arguments: runs every defense found under defenses/*/defend.sh on
# the public slice in data/inria_slice, scores it with the protected detector and writes
# scoring/scores_train.json.  Inputs, defended outputs, detections and per-defense logs
# are kept in outputs/eval_train/ for inspection.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scoring
python eval/train/evaluate.py
