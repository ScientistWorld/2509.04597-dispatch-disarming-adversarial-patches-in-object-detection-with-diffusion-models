#!/bin/bash
# Baselines.  The undefended detector ("vanilla") is computed by the evaluator itself on
# the same inputs as every defense.  The paper's two ablation baselines are runnable but
# not registered under defenses/ (on the hidden test both lower clean mAP.5 by more than
# 0.05 and would be scored as failed):
#   method/ablations/regenerated_only.sh   regenerated image, no rectification (Appendix A.1)
#   method/ablations/dispatch_gray.sh      DISPATCH(G), gray fill instead of regeneration (A.2)
# This script scores them on the public slice (writes outputs/baseline_scores.json).
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=outputs/baseline_defenses
rm -rf "$TMP"
mkdir -p "$TMP/regenerated_only" "$TMP/dispatch_gray"
cp method/ablations/regenerated_only.sh "$TMP/regenerated_only/defend.sh"
cp method/ablations/dispatch_gray.sh "$TMP/dispatch_gray/defend.sh"
python eval/train/evaluate.py --defenses-dir "$TMP" --score-path outputs/baseline_scores.json --save-dir outputs/eval_baselines
