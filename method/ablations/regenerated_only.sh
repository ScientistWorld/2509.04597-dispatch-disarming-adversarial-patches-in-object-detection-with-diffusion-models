#!/bin/bash
# Paper ablation (Appendix A.1): the fully regenerated image, without rectification.
set -euo pipefail
python method/dispatch.py --input-dir "$1" --output-dir "$2" --mode regen
