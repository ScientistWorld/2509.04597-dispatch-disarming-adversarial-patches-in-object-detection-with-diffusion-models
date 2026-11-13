#!/bin/bash
# End-to-end reproduction.  The benchmark assets (data/inria_slice, eval/test/assets) are
# committed; rebuild them only with eval/train/build_slice.py / eval/test/build_assets.py.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/download.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
