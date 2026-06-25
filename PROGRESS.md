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

Artifact inspection showed the hiding failure came from noisy diffusion regeneration, not from the evaluator. The sampler was using a one-step DDPM update across skipped timesteps, and training only used local block masks while inference uses complementary checkerboards. The current code fixes both issues and reruns the benchmark.

## Remaining

- Rerun the compute job after the sampler/training-mask fixes.
- Inspect the new `scoring/scores.json` and decide whether the scaled reproduction supports `core_claim` or needs another tuning pass.
- If hiding recovery improves, expand to an additional setting; otherwise inspect regenerated images and mask coverage again.

## Issues

- The original full INRIA/APRICOT plus MMDetection plus pretrained LDM setup is too large for a first-turn reproduction. This workspace starts with a scaled benchmark that preserves the algorithmic computation.
- First run exposed an implementation-level diffusion issue: skipped-timestep sampling produced noisy regenerated images, causing KMeans masks to cover benign object structure.

## Deviations from Paper

- The paper uses INRIA-Person/APRICOT and COCO-pretrained LDM inpainting. This reproduction uses a synthetic object-detection dataset and trains a small DDPM inpainting model on clean synthetic images to keep the first experiment feasible without large external downloads.
- The paper evaluates YOLOv3, Faster R-CNN, DETR, SSD, and RetinaNet. This reproduction uses a fixed method-agnostic detector in `eval/` so the benchmark can run quickly and future methods can be compared consistently.
- The paper reports mAP.5 and AR. The scaled hiding experiment maps recall at IoU 0.5 into the corresponding score fields because the synthetic detector does not emit a full ranked detection set suitable for COCO-style AP.
