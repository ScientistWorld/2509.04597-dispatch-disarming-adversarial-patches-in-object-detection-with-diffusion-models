# Verification Report (hardening iteration)

- Final job `eb6dbd07-3f3`: `scripts/evaluate_train.sh`, `evaluate_test.sh` and
  `evaluate.sh` run with no arguments and reproduce the reference scores exactly
  (deterministic seeds); `python validate.py --post-design` passes (soft warnings only
  about RNG use in the benchmark builders).
- Tampering with one pinned hidden-test source image makes every defense fail.
- Shortcut probes, variants and seeds were run through both evaluators in job
  `ae674dfc-9b7`; results and replayed ranks are in `redteam/hardening/results/` and
  summarised in `HARDENING.md`.

- Milestone: secondary_claims
- Ready for gym use: yes (hidden test needs one GPU)
- Key limitation: single protected detector; simulated physical creating patches.
