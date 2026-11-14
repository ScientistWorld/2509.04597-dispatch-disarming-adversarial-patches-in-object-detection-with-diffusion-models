#!/bin/bash
# Paper ablation DISPATCH(G) (Appendix A.2): the detected adversarial region is
# replaced by gray pixels instead of regenerated content ("detect and remove").
set -euo pipefail
python method/dispatch.py --input-dir "$1" --output-dir "$2" --mode gray
