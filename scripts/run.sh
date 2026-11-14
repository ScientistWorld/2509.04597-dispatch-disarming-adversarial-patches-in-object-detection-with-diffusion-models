#!/bin/bash
# Reference reproduction + final verification job (1 GPU, ~20 min).
#   1. scripts/evaluate_train.sh, evaluate_test.sh, evaluate.sh with no arguments
#      -> scoring/scores_train.json, scores_test.json, scores.json (reference defenses only)
#   2. tampered fixed input demo: one hidden-test source image is altered; the hidden
#      evaluator must score every defense as failed (then the file is restored)
#   3. the paper's two ablations scored without the clean gate, from saved detections
#   4. python validate.py --post-design
# The full hardening job (asset build + every probe) is redteam/hardening/run_hardening.sh.
set -uo pipefail
cd /home/user
chmod -R u+rwX eval/test data/downloads/INRIAPerson data/downloads/torchvision 2>/dev/null || true
rm -rf outputs/.eval_t*
# Restore a fixed input left altered by an interrupted tamper demo.
if [ -f tmp/tamper_backup.png ] && [ -f tmp/tamper_src ]; then cp tmp/tamper_backup.png "$(cat tmp/tamper_src)"; rm -f tmp/tamper_backup.png tmp/tamper_src; fi
R=redteam/hardening/results
mkdir -p "$R" tmp
python -c "import torch; [print('cuda', i, torch.cuda.get_device_name(i), 'free GB %.1f' % (torch.cuda.mem_get_info(i)[0] / 1e9)) for i in range(torch.cuda.device_count())]"
T0=$(date +%s)
el() { echo "[elapsed $(( $(date +%s) - T0 ))s] $*"; }

el "evaluate_train"; bash scripts/evaluate_train.sh > outputs/final_train.log 2>&1 || echo "evaluate_train FAILED"
el "evaluate_test";  bash scripts/evaluate_test.sh  > outputs/final_test.log 2>&1  || echo "evaluate_test FAILED"
cp outputs/eval_test/status.json $R/status_test_final.json
el "evaluate";       bash scripts/evaluate.sh       > outputs/final_main.log 2>&1  || echo "evaluate FAILED"

el "tamper demo"
SRC=$(python -c "import json; it=json.load(open('eval/test/assets/spec.json'))['items'][0]; print('data/downloads/INRIAPerson/%s/pos/%s' % (it['split'], it['source']))")
cp "$SRC" tmp/tamper_backup.png; echo "$SRC" > tmp/tamper_src
rm -rf tmp/tamper_defenses; mkdir -p tmp/tamper_defenses/identity
printf '#!/bin/bash\nset -e\ncp "$1"/*.png "$2"/\n' > tmp/tamper_defenses/identity/defend.sh
python - "$SRC" <<'EOF'
import sys
from PIL import Image
p = sys.argv[1]
im = Image.open(p).convert("RGB")
im.paste((255, 0, 0), (0, 0, 40, 40))   # visibly altered fixed input
im.save(p)
EOF
python eval/test/evaluate.py --defenses-dir tmp/tamper_defenses --score-path $R/tamper_demo_scores.json \
  --log-dir outputs/tamper_demo > outputs/tamper_demo.log 2>&1 || echo "tamper demo FAILED"
cp tmp/tamper_backup.png "$SRC" && rm -f tmp/tamper_backup.png tmp/tamper_src
cp outputs/tamper_demo/status.json $R/tamper_demo_status.json
grep "FIXED INPUTS" outputs/tamper_demo.log; cat $R/tamper_demo_status.json

el "ungated ablations"
python - <<'EOF'
import json, sys
sys.path.insert(0, "eval/test")
from common import score_predictions  # analysis only: rescoring saved detections without the gate
out = {}
for name, path in (("hidden_test", "outputs/ref_test_predictions.json"), ("public_slice", "outputs/ref_train_predictions.json")):
    try:
        d = json.load(open(path))
    except OSError:
        continue
    out[name] = {m: score_predictions(d["items"], p) for m, p in d["predictions"].items()
                 if m in ("vanilla", "dispatch", "dispatch_gray", "regenerated_only")}
json.dump(out, open("redteam/hardening/results/ungated_ablations.json", "w"), indent=1)
print(json.dumps(out, indent=1))
EOF

el "validate"
python validate.py --post-design 2>&1 | tail -30
chmod -R u+rwX eval/test data/downloads/INRIAPerson data/downloads/torchvision 2>/dev/null || true
el "done"
python - <<'EOF'
import json
for f in ("scoring/scores_test.json", "scoring/scores_train.json"):
    s = json.load(open(f)); print("=====", f)
    for exp, d in s["experiments"].items():
        for m, r in d["results"].items():
            print(f"  {exp[:22]:22s} {m:10s}", " ".join(f"{k}={v:.3f}" for k, v in r.items()))
EOF
echo FINAL_DONE
