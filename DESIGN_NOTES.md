# Train/Test Evaluation Design Notes

- One contract for both slices: `bash defenses/<name>/defend.sh <in> <out>`; the
  evaluators discover methods on disk (no method name is hardcoded, nothing is read
  from `scores_train.json`).
- `eval/train/` and `eval/test/` share no code; `eval/` never imports `method/`.
- Both slices are drawn from the same pool (902 INRIA-Person positives); the public
  slice takes 200 by a public hash, the hidden test 480 of the rest by a hidden salt.
  Each slice's patches are optimised on images it does not evaluate.
- Every source image is used for exactly one variant (clean / hide / hide_large /
  create), so no attacked input has a clean counterpart; variants are mixed under
  opaque ids in one input directory.
- Hidden inputs are composed at test time; while a defense runs, `eval/test/`, the
  INRIA directory and the detector weights are unreadable.  Fixed inputs are pinned by
  SHA-256; a mismatch fails every defense.
- Failures (exit code, timeout, missing / unreadable / mis-sized outputs, evaluator
  exceptions) and constraint violations (clean or object person mAP.5 more than 0.05
  below no defense) give the worst value in every metric of every experiment.
- Unannotated people (INRIA is incomplete) become ignore regions from confident
  clean-image detections, so train and test clean mAP.5 agree (0.980 / 0.976).
