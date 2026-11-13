#!/bin/bash
# Run the paper's method (DISPATCH) on a directory of PNG images.
#   bash scripts/method.sh <input_dir> <output_dir> [extra method/dispatch.py flags]
# Defaults: the public slice inputs written by scripts/evaluate.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-outputs/eval/inputs}"
OUT="${2:-outputs/dispatch_demo}"
shift $(( $# > 2 ? 2 : $# ))
python method/dispatch.py --input-dir "$IN" --output-dir "$OUT" "$@"
