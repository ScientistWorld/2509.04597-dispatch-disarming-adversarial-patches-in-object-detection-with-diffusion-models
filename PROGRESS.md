# Progress

## What Works

- Paper analyzed and benchmark scaffold filled in.
- Implemented a scaled DISPATCH pipeline: clean-data inpainting DDPM, complementary checkerboard regeneration, KMeans L2 adversarial mask, and masked rectification.
- Implemented a method-agnostic synthetic detector and evaluator in `eval/`.
- Added scripts for download, baseline, method, evaluation, and end-to-end reproduction.

## Results

No compute job has completed yet. The next submitted job will train the scaled inpainting model and write `scoring/scores.json`.

## Remaining

- Run the first compute job to verify the container, training loop, DISPATCH inference, and evaluation.
- Inspect `scoring/scores.json` and decide whether the scaled reproduction supports `core_claim` or needs tuning.
- Commit job results and update milestones after the job returns.

## Issues

- The original full INRIA/APRICOT plus MMDetection plus pretrained LDM setup is too large for a first-turn reproduction. This workspace starts with a scaled benchmark that preserves the algorithmic computation.

## Deviations from Paper

- The paper uses INRIA-Person/APRICOT and COCO-pretrained LDM inpainting. This reproduction uses a synthetic object-detection dataset and trains a small DDPM inpainting model on clean synthetic images to keep the first experiment feasible without large external downloads.
- The paper evaluates YOLOv3, Faster R-CNN, DETR, SSD, and RetinaNet. This reproduction uses a fixed method-agnostic detector in `eval/` so the benchmark can run quickly and future methods can be compared consistently.
- The paper reports mAP.5 and AR. The scaled hiding experiment maps recall at IoU 0.5 into the corresponding score fields because the synthetic detector does not emit a full ranked detection set suitable for COCO-style AP.
