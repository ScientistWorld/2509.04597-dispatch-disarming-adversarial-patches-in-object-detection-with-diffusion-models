#!/bin/bash
# Full hardening job (1 GPU):
#   1. build the visible slice (data/inria_slice) and the hidden test assets (eval/test/assets)
#   2. reference runs: scripts/evaluate_train.sh, evaluate_test.sh, evaluate.sh
#   3. hardening probes (better/weaker variants, seeds, simple baselines, shortcuts) through
#      the hidden-test and visible-slice evaluators
#   4. restore the reference score files (some shortcut probes deliberately overwrite them)
# Stages are resumable: finished stages leave markers in outputs/.
set -uo pipefail
cd /home/user
chmod -R u+rwX eval/test data/downloads/INRIAPerson/Test data/downloads/torchvision 2>/dev/null || true
rm -rf outputs/.eval_t* tmp/smoke
mkdir -p outputs redteam/hardening/results
R=redteam/hardening/results
nvidia-smi --query-gpu=name,memory.total --format=csv || true
T0=$(date +%s)
el() { echo "[elapsed $(( $(date +%s) - T0 ))s] $*"; }

# 1. Assets ---------------------------------------------------------------------------
if [ ! -f data/inria_slice/spec.json ]; then
  el "build visible slice"
  python eval/train/build_slice.py 2>&1 | tee $R/build_slice.log
fi
if [ ! -f eval/test/assets/spec.json ]; then
  el "build hidden test assets"
  python eval/test/build_assets.py > outputs/build_assets.log 2>&1
  grep -E "it=(0|2999|599) |wrote" outputs/build_assets.log
fi
[ -f data/inria_slice/spec.json ] && [ -f eval/test/assets/spec.json ] || { echo "ASSET BUILD FAILED"; exit 1; }

# 2. Reference runs ------------------------------------------------------------------
if [ ! -f outputs/.stage_reference.done ]; then
  el "reference: evaluate_train"
  bash scripts/evaluate_train.sh > outputs/ref_train.log 2>&1 || echo "evaluate_train FAILED"
  cp outputs/eval_train/status.json $R/status_train_reference.json
  el "reference: evaluate_test"
  bash scripts/evaluate_test.sh > outputs/ref_test.log 2>&1 || echo "evaluate_test FAILED"
  cp outputs/eval_test/status.json $R/status_test_reference.json
  el "reference: evaluate (scores.json)"
  bash scripts/evaluate.sh > outputs/ref_main.log 2>&1 || echo "evaluate FAILED"
  for f in scores_train scores_test scores; do cp scoring/$f.json $R/reference_$f.json; done
  touch outputs/.stage_reference.done
fi
cat $R/status_test_reference.json

# 3. Probes ------------------------------------------------------------------------------
cp data/downloads/torchvision/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth redteam/hardening/defenses/sc_detector_guided/weights.pth
if [ ! -f outputs/.stage_probes_test.done ]; then
  el "probes: hidden test"
  python eval/test/evaluate.py --defenses-dir redteam/hardening/defenses \
    --score-path $R/scores_test_probes.json --log-dir outputs/hardening_test > outputs/probes_test.log 2>&1 || echo "probe test FAILED"
  cp outputs/hardening_test/status.json $R/status_test_probes.json
  for f in outputs/hardening_test/logs/sc_*.log outputs/hardening_test/logs/var_*.log; do echo "== $f"; grep -v "making attention\|Working with z" "$f" | tail -2; done
  touch outputs/.stage_probes_test.done
fi
if [ ! -f outputs/.stage_probes_train.done ]; then
  el "probes: visible slice"
  python eval/train/evaluate.py --defenses-dir redteam/hardening/defenses \
    --score-path $R/scores_train_probes.json --save-dir outputs/hardening_train > outputs/probes_train.log 2>&1 || echo "probe train FAILED"
  cp outputs/hardening_train/status.json $R/status_train_probes.json
  touch outputs/.stage_probes_train.done
fi
rm -f redteam/hardening/defenses/sc_detector_guided/weights.pth

# 4. Restore reference score files ---------------------------------------------------------
for f in scores_train scores_test scores; do cp $R/reference_$f.json scoring/$f.json; done
chmod -R u+rwX eval/test data/downloads/INRIAPerson/Test data/downloads/torchvision 2>/dev/null || true
el "done"
python - <<'EOF'
import json
for name in ("reference_scores_test", "scores_test_probes", "reference_scores_train", "scores_train_probes"):
    s = json.load(open(f"redteam/hardening/results/{name}.json"))
    print("=====", name)
    for exp, d in s["experiments"].items():
        print("==", exp)
        for m, r in d["results"].items():
            print(f"  {m:22s}", " ".join(f"{k}={v:.3f}" for k, v in r.items()))
EOF
echo FULL_DONE
