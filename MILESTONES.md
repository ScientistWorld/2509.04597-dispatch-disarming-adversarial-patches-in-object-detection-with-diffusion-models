# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification
- Not stopped.
