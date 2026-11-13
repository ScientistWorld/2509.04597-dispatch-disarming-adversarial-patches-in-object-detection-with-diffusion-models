#!/bin/bash
# Hidden test.  No arguments: runs every defense found under defenses/*/defend.sh on
# inputs composed at test time from eval/test/assets (never shown to methods), runs
# the protected detector itself, and writes scoring/scores_test.json.
# Needs one GPU (the reference defense regenerates every image with a diffusion model);
# about 3 minutes per diffusion-based defense on an A100.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scoring
python eval/test/evaluate.py
