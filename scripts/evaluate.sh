#!/bin/bash
# Standard evaluation entry point: scores every defense under defenses/ on the public
# INRIA-Person slice (data/inria_slice) and writes scoring/scores.json.
# Uses only code in eval/; defenses are run through their defend.sh contract
# (see defenses/README.md).  Artifacts go to outputs/eval/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scoring
python eval/train/evaluate.py --score-path scoring/scores.json --save-dir outputs/eval
