# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-25 00:00] - none
- Read the paper and identified DISPATCH's core computational steps.
- Implemented a scaled reproduction environment preserving the regenerate-and-rectify algorithm.
- Prepared a compute job to train and evaluate the implementation.

### [2026-06-25 05:03] - method_runs
- Completed the first compute job: generated synthetic data, trained the clean-data inpainting DDPM, ran vanilla and DISPATCH pipelines, and wrote `scoring/scores.json`.
- Confirmed creating-patch suppression works on the scaled benchmark, but hiding-patch recovery fails because diffusion regeneration was noisy.
- Fixed the sampler/training mismatch and prepared a second job to rerun the benchmark.

### [2026-06-25 05:11] - method_runs
- Completed the second compute job with skipped-timestep sampling and checkerboard-training fixes.
- Creating-patch suppression remained strong, but hiding-patch recovery still failed because regenerated images were visibly noisy.
- Switched the compact diffusion model to `x0` prediction and prepared another run.

### [2026-06-25 05:19] - method_runs
- Completed the third compute job with `x0` diffusion prediction.
- Hiding recovery improved to 0.580 mAP.5/AR, but remained below the vanilla attack score of 0.696; creating-patch suppression regressed.
- Strengthened the synthetic hiding attack, increased training, and switched to finer checkerboard regeneration for the next run.

### [2026-06-25 05:41] - core_claim
- Completed the fourth compute job with stronger synthetic hiding patches and longer clean-data inpainting training.
- DISPATCH recovered hiding attack mAP.5/AR to 0.545 while vanilla detection under the hiding attack was 0.000; clean-image performance was measured at 0.750.
- Identified and fixed a method-agnostic creating-attack evaluator bug that counted true-person detections overlapping the patch as false patch detections; prepared an evaluation-only job to rewrite `scoring/scores.json`.

### [2026-06-25 05:44] - core_claim_plus
- Completed the evaluation-only job with the corrected creating-attack metric.
- Confirmed DISPATCH improves both scaled hiding recovery (0.545 vs 0.000 attack mAP.5/AR) and scaled creating suppression (0.250 vs 0.325 ASR).
- Restored `scripts/run.sh` and `scripts/method.sh` to the successful full reproduction defaults.

### [2026-06-25 05:47] - core_claim_plus
- Fixed the scoring schema so reproduced scaled-synthetic results are reported as `synthetic_hiding_scaled` and `synthetic_creating_scaled`, not as the paper's full INRIA/APRICOT rows.
- Re-ran `python validate.py --compare`; all checks passed.

## Stop Justification
- Completed at milestone `core_claim_plus`.
