#!/bin/bash
# Baselines.  The undefended detector ("vanilla") is computed by the evaluator itself on
# the same inputs; the paper's two ablation baselines are registered defenses:
#   defenses/regenerated_only  (Appendix A.1, no rectification)
#   defenses/dispatch_gray     (Appendix A.2, DISPATCH(G): gray fill instead of regeneration)
# They are scored together with every other defense by the evaluation scripts.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/evaluate.sh
