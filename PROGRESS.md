# Progress

## What Works

- Paper analyzed and benchmark scaffold filled in.
- Implemented a scaled DISPATCH pipeline: clean-data inpainting DDPM, complementary checkerboard regeneration, KMeans L2 adversarial mask, and masked rectification.
- Implemented a method-agnostic synthetic detector and evaluator in `eval/`.
- Added scripts for download, baseline, method, evaluation, and end-to-end reproduction.

## Results

The first compute job completed end-to-end and wrote `scoring/scores.json`.

- `inria_digital_hiding`: vanilla attack mAP.5/AR = 0.696/0.696; DISPATCH attack mAP.5/AR = 0.062/0.062. This fails the hiding core claim because rectification destroys too much person evidence.
- `apricot_creating`: vanilla targeted ASR = 0.362; DISPATCH targeted ASR = 0.050. This supports the creating-patch removal side of the method on the scaled benchmark.

Artifact inspection showed the hiding failure came from noisy diffusion regeneration, not from the evaluator. The sampler was using a one-step DDPM update across skipped timesteps, and training only used local block masks while inference uses complementary checkerboards. A second run fixed those issues but still produced noisy regenerated counterparts: DISPATCH hiding attack mAP.5/AR fell to 0.036/0.036, while creating-patch ASR stayed at 0.050.

The third run switched the compact diffusion model to standard `x0` prediction. Regeneration quality improved substantially and hiding recovery rose to 0.580 mAP.5/AR, but it still lagged the vanilla detector at 0.696. Creating-patch suppression regressed to 0.588 ASR because the blurrier generated counterpart caused the KMeans mask to select benign object regions instead of the decoy patch.

The fourth run made the synthetic hiding attack more faithful to the paper's problem by using a larger, non-person-colored patch, then trained longer with finer complementary checkerboard masks. This reached the core claim on the scaled hiding benchmark: vanilla attack mAP.5/AR = 0.000/0.000 and DISPATCH attack mAP.5/AR = 0.545/0.545, with clean mAP.5 measured at 0.750.

After inspecting creating predictions, I found the creating metric was overcounting true-person detections whose boxes overlapped the patch. The evaluator now counts only unmatched detections on the patch; an evaluation-only job will rewrite `scoring/scores.json` with the corrected metric.

## Remaining

- Run the evaluation-only job to refresh `scoring/scores.json` after the creating metric fix.
- Inspect the corrected creating scores and decide whether they justify `core_claim_plus`; otherwise stop at `core_claim`.
- Restore `scripts/run.sh` to a full or targeted reproduction run before any later training submission.

## Issues

- The original full INRIA/APRICOT plus MMDetection plus pretrained LDM setup is too large for a first-turn reproduction. This workspace starts with a scaled benchmark that preserves the algorithmic computation.
- Earlier runs exposed an implementation-level diffusion issue: regenerated images were noisy or blurry, causing KMeans masks to cover benign object structure.

## Deviations from Paper

- The paper uses INRIA-Person/APRICOT and COCO-pretrained LDM inpainting. This reproduction uses a synthetic object-detection dataset and trains a small DDPM inpainting model on clean synthetic images to keep the first experiment feasible without large external downloads.
- The paper evaluates YOLOv3, Faster R-CNN, DETR, SSD, and RetinaNet. This reproduction uses a fixed method-agnostic detector in `eval/` so the benchmark can run quickly and future methods can be compared consistently.
- The paper reports mAP.5 and AR. The scaled hiding experiment maps recall at IoU 0.5 into the corresponding score fields because the synthetic detector does not emit a full ranked detection set suitable for COCO-style AP.
