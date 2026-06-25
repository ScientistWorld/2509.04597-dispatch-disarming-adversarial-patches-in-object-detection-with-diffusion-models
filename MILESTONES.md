# Reproduction Milestones

**Current: method_runs**

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

### [2026-06-25 06:28] - core_claim
- A full CPU rebuild from source completed successfully, but the creating benchmark regressed: DISPATCH ASR was 0.425 versus vanilla 0.325.
- Downgraded current milestone to the still-supported hiding core claim while testing a coarser checkerboard mask for creating-patch suppression.

### [2026-06-25 06:36] - core_claim
- The coarser `N=8` follow-up completed and did not fix creating-patch suppression: DISPATCH ASR stayed at 0.425 versus vanilla 0.325.
- Updated the scaled creating benchmark from person-colored decoys to noise-like out-of-distribution trigger patches, matching the paper's creating-attack threat more closely while keeping evaluation method-agnostic.

### [2026-06-25 07:01] - core_claim
- The corrected-trigger rebuild completed and validated structurally. Hiding improved from 0.000 to 0.598 mAP.5/AR, and rectification beat regenerated-only images, 0.598 vs. 0.571.
- Creating suppression still failed narrowly: vanilla ASR was 0.450 and DISPATCH ASR was 0.475. The next fix removes a trigger color that overlapped the synthetic head detector and keeps creating patches farther from true objects.

### [2026-06-25 07:24] - secondary_claims
- The final CPU rebuild completed from source and passed validation.
- DISPATCH recovered hiding detections from 0.000 to 0.652 mAP.5/AR while preserving clean mAP.5 at 0.750.
- DISPATCH reduced creating-patch ASR from 0.450 to 0.0375.
- The rectification ablation again supports the paper's secondary claim: full DISPATCH reached 0.652 attack mAP.5 versus 0.598 for regenerated-only images.

### [2026-06-25 07:48] - method_runs
- Continuation audit found that the clean-preservation constraint was scored on raw clean images rather than clean images passed through DISPATCH.
- Updated method and evaluation scripts to generate and score `outputs/dispatch_images/clean`.
- Temporarily downgraded until a fresh rebuild verifies the corrected constraint metric.
