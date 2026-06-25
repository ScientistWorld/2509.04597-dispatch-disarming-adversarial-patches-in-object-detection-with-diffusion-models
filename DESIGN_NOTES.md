# Train/Test Evaluation Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. Train invokes `eval/train/evaluate.py` and writes `scoring/scores_train.json`; test invokes `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked for hardcoded method defaults. The split scripts do not pin a method name; evaluators discover artifact folders and the test side can mirror method keys from `scores_train.json`.
- Checked leakage with grep. There are no imports between `eval/train/` and `eval/test/`.
- Checked for paper-agreement metric names. No nonzero metric is named as deviation from paper or fit to published results.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both split scripts. No shell syntax errors.

## Changed

- Fixed a metric shortcut in both train and test evaluators: person-object recall now filters detections to `class == "person"` before matching ground-truth person boxes. Previously a synthetic trigger component could count toward person recovery if it overlapped a person box.
- Replaced duplicate hiding `attack_AR` with mean recall over IoU thresholds 0.5, 0.6, and 0.7. This makes it harder to game the visible metric with loose boxes that only satisfy IoU 0.5.
- Added `attack_precision` and `clean_precision` for the hiding split. These penalize methods that raise recall by hallucinating extra people, including on clean images.
- Split the creating metrics across multiple patch-overlap thresholds instead of emitting four copies of the same false-positive rate. This gives train/test more ways to distinguish broad patch suppression from narrow threshold tuning.
- Added `object_mAP5` to the creating split so a method cannot improve creating ASR by erasing all objects.
- Added `attack_AR` and `attack_precision` to the rectification ablation so direct-regeneration versus rectification is tested on localization quality and false positives, not just one recall threshold.
- Updated `scoring/reference.json` metric definitions with descriptions and coefficients for the added metrics. Existing paper-reported result values were not changed.

## Runtime Concerns For Runner

- This iteration did not run Python evaluators or generate numbers by instruction. The next runner should execute both split scripts and confirm the new metrics appear in `scores_train.json` and `scores_test.json`.
- Existing synthetic reference result rows do not include values for the newly added metrics because no scoring run was allowed here. If the validator or report tooling expects complete reference rows, regenerate the split scores and refresh only the scaled synthetic reference rows with measured values.
- The synthetic creating `targeted_*` and `untargeted_*` metrics are thresholded patch-overlap proxies, not true class-targeted detector metrics. They are intentionally different thresholds over the same saved image artifacts to reduce threshold overfitting.
- The split still uses the existing synthetic test images divided by stratified hash. Sample count is limited, so watch for metric saturation or high variance when the runner produces the first train/test numbers.
