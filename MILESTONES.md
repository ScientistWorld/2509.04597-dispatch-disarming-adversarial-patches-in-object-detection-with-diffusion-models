# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-06-25 04:58] - none
- Started from a partially initialized DISPATCH reproduction workspace.
- Identified the paper's core regenerate-and-rectify defense for adversarial patches in object detection.

### [2026-06-25 05:12] - method_runs
- Implemented the scaled DISPATCH pipeline: clean-data inpainting DDPM, complementary checkerboard regeneration, L2 discrepancy masking, and masked rectification.
- Added synthetic hiding and creating attack data generation plus method-agnostic evaluation.

### [2026-06-25 05:44] - core_claim
- Ran the proposed method end-to-end on the scaled hiding benchmark.
- DISPATCH improved attack mAP.5 from 0.000 for vanilla attacked images to 0.545 on rectified images.

### [2026-06-25 06:05] - core_claim_plus
- Added the scaled creating-attack benchmark.
- DISPATCH reduced patch-triggered ASR from 0.325 to 0.250 while preserving the output-based evaluation contract.

### [2026-06-25 06:24] - secondary_claims
- Added the rectification ablation matching the paper's Appendix A.1 claim.
- Full DISPATCH rectification improved attack mAP.5 over regenerated-only images, 0.545 vs. 0.491, on the scaled benchmark.
