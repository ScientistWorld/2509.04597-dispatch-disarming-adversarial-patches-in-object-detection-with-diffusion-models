#!/bin/bash
# Job script (MODE=smoke): end-to-end smoke test of the rebuilt gym on a GPU node.
# Tiny patch budgets and 6 items per variant; everything is written under tmp/smoke/.
set -uo pipefail
cd /home/user
# Self-heal permissions in case an interrupted hidden-test run left locks behind.
chmod -R u+rwX eval/test data/downloads/INRIAPerson/Test data/downloads/torchvision 2>/dev/null || true

S=tmp/smoke
rm -rf "$S"; mkdir -p "$S/defenses"
nvidia-smi || true
python -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

for d in defenses/dispatch defenses/regenerated_only defenses/dispatch_gray \
         redteam/hardening/defenses/var_iter2 redteam/hardening/defenses/simple_blur \
         redteam/hardening/defenses/sc_blank redteam/hardening/defenses/sc_partial \
         redteam/hardening/defenses/sc_read_answers redteam/hardening/defenses/sc_template_visible \
         redteam/hardening/defenses/sc_crash_forge; do
  ln -s "/home/user/$d" "$S/defenses/$(basename $d)"
done

T0=$(date +%s)
echo "== build visible slice (smoke)"
python eval/train/build_slice.py --out-dir "$S/slice" --hide-iters 30 --create-iters 30 --per-variant 6 || echo "BUILD_SLICE_FAILED"
echo "== build hidden assets (smoke)"
python eval/test/build_assets.py --out-dir "$S/assets" --hide-iters 30 --create-iters 30 --per-variant 6 || echo "BUILD_ASSETS_FAILED"
echo "elapsed $(( $(date +%s) - T0 ))s"

echo "== train evaluator (smoke)"
python eval/train/evaluate.py --slice-dir "$S/slice" --defenses-dir "$S/defenses" \
  --score-path "$S/scores_train.json" --save-dir "$S/eval_train" || echo "EVAL_TRAIN_FAILED"
echo "elapsed $(( $(date +%s) - T0 ))s"

echo "== DISPATCH debug images"
python method/dispatch.py --input-dir "$S/eval_train/inputs" --output-dir "$S/debug_out" --debug-dir "$S/debug" || echo "DEBUG_FAILED"

echo "== test evaluator (smoke)"
python eval/test/evaluate.py --assets-dir "$S/assets" --defenses-dir "$S/defenses" \
  --score-path "$S/scores_test.json" --log-dir "$S/eval_test" || echo "EVAL_TEST_FAILED"
ls -ld eval/test
echo "elapsed $(( $(date +%s) - T0 ))s"
python -c "import torch; print('peak GPU mem GB', torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0)"
cat "$S"/eval_train/logs/dispatch.log | tail -5
cat "$S"/eval_test/logs/sc_read_answers.log | tail -3
echo SMOKE_DONE
